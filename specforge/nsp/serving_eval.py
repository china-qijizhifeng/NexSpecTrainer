"""Request-scoped AL and bounded concurrent serving measurements."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import uuid

import requests

from specforge.nsp.metrics import MetricTotals, parse_bins
from specforge.nsp.store import atomic_json


def serving_terms(meta, completion_tokens, grouped):
    cycles = meta.get("spec_verify_cycles")
    if cycles is not None:
        if not isinstance(cycles, list) or not cycles:
            raise ValueError("spec_verify_cycles must contain observed verify cycles")
        seen, rows = set(), []
        for cycle in cycles:
            cycle_id = cycle["cycle_id"]
            if cycle_id in seen:
                raise ValueError("duplicate verify cycle ID")
            seen.add(cycle_id)
            ctx, emitted = cycle["ctx_len"], cycle["committed_output_tokens"]
            if any(type(v) is not int or v < 0 for v in (ctx, emitted)):
                raise ValueError("verify cycle counts must be non-negative integers")
            rows.append((ctx, emitted, 1))
        if sum(row[1] for row in rows) != completion_tokens:
            raise ValueError("verify-cycle output total disagrees with completion_tokens")
        return rows
    if grouped:
        raise ValueError("dynamic ctx bins require spec_verify_cycles telemetry; request-level AL is insufficient")
    verifies = meta.get("spec_verify_ct")
    if type(verifies) is not int or verifies <= 0:
        raise ValueError("missing positive spec_verify_ct; cannot measure AL")
    if type(completion_tokens) is not int or completion_tokens <= 0:
        raise ValueError("missing positive completion_tokens")
    return [(None, completion_tokens, verifies)]


def _identity(serving, info):
    args = info.get("server_args", info)
    keys = {serving.checkpoint_field, "model_path", "weight_version", "speculative_algorithm",
            "speculative_num_draft_tokens", "json_model_override_args", "disable_radix_cache"}
    return {key: args.get(key) for key in sorted(keys | set(serving.expected_server_args))}


def _request(spec, prompt, index, headers, phase):
    serving = spec.serving
    request_id = "nsp-" + uuid.uuid4().hex
    sampling = {"temperature": serving.temperature, "top_p": serving.top_p,
                "max_new_tokens": serving.max_new_tokens, "seed": spec.seed + index}
    if serving.protocol == "native":
        ids = prompt.get("input_ids")
        if not isinstance(ids, list) or not ids or any(type(v) is not int or v < 0 for v in ids):
            raise ValueError("native prompts require nonempty nonnegative input_ids")
        path = "/generate"
        body = {"rid": request_id, "input_ids": ids, "stream": False, "sampling_params": sampling}
    else:
        if not isinstance(prompt.get("messages"), list) or not prompt["messages"]:
            raise ValueError("each prompt must contain a nonempty messages array")
        path = "/v1/chat/completions"
        body = {"model": serving.model, "messages": prompt["messages"], "stream": False,
                "temperature": serving.temperature, "top_p": serving.top_p,
                "max_tokens": serving.max_new_tokens, "seed": spec.seed + index}
    begin = time.perf_counter()
    response = requests.post(serving.url.rstrip("/") + path, json=body,
                             headers={**headers, "X-Request-Id": request_id}, timeout=serving.timeout)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise ValueError(f"serving error: {payload['error']}")
    duration = time.perf_counter() - begin
    if serving.protocol == "native":
        meta = payload["meta_info"]
        tokens = meta["completion_tokens"]
        output = payload.get("output_ids")
        if not isinstance(output, list) or len(output) != tokens or any(type(v) is not int or v < 0 for v in output):
            raise ValueError("native output token IDs disagree with completion_tokens")
    else:
        meta = payload["choices"][0].get("meta_info", payload.get("meta_info", {}))
        tokens = payload["usage"]["completion_tokens"]
        output = None
    if type(tokens) is not int or not 0 < tokens <= serving.max_new_tokens:
        raise ValueError("invalid completion token count")
    rows = serving_terms(meta, tokens, bool(spec.group_by)) if "al" in spec.metrics else []
    return {"request_id": request_id, "sample_id": prompt.get("id", index), "phase": phase,
            "input_ids": prompt.get("input_ids"), "output_ids": output,
            "completion_tokens": tokens, "duration_seconds": duration, "verify_rows": rows,
            "meta_info": meta, "response": payload}


def run_serving(spec, attempt):
    serving, attempt = spec.serving, Path(attempt)
    from specforge.nsp.eval_inputs import serving_checkpoint_identity

    checkpoint_identity = serving_checkpoint_identity(spec.checkpoint)
    recorded_identity = attempt / "checkpoint-identity.json"
    if recorded_identity.exists() and json.loads(recorded_identity.read_text()) != checkpoint_identity:
        raise ValueError("serving checkpoint metadata changed after submission")
    headers = {}
    if serving.api_key_env:
        token = os.environ.get(serving.api_key_env)
        if not token:
            raise ValueError(f"missing credential environment variable {serving.api_key_env}")
        headers["Authorization"] = f"Bearer {token}"
    def server_info():
        response = requests.get(serving.url.rstrip("/") + serving.info_path,
                                headers=headers, timeout=serving.timeout)
        response.raise_for_status()
        return response.json()
    info = server_info()
    arguments = info.get("server_args", info)
    if arguments.get(serving.checkpoint_field) != spec.checkpoint:
        raise ValueError("serving endpoint does not attest the requested checkpoint; use a dedicated matching server")
    for key, expected in serving.expected_server_args.items():
        if arguments.get(key) != expected:
            raise ValueError(f"serving endpoint argument mismatch: {key}")
    prompt_bytes = Path(serving.prompts).read_bytes()
    prompts = [json.loads(line) for line in prompt_bytes.splitlines() if line.strip()]
    if not prompts:
        raise ValueError("prompt set is empty")
    totals = MetricTotals(["al"] if "al" in spec.metrics else [], parse_bins(spec.ctx_bins) if spec.group_by else None)
    receipts = []
    wall = None
    result = {"schema_version": 1, "stage": "serving", "complete": False,
              "checkpoint": spec.checkpoint, "runtime": {"version": info.get("version"), **_identity(serving, info)},
              "prompts_sha256": hashlib.sha256(prompt_bytes).hexdigest()}
    try:
        with (attempt / "warmup.jsonl").open("x") as warmup:
            for index in range(serving.warmup_requests):
                row = _request(spec, prompts[index % len(prompts)], index, headers, "warmup")
                warmup.write(json.dumps(row) + "\n")
                warmup.flush()
        begin = time.perf_counter()
        with (attempt / "requests.jsonl").open("x") as stream, ThreadPoolExecutor(max_workers=serving.concurrency) as pool:
            futures = [pool.submit(_request, spec, prompt, index, headers, "measure") for index, prompt in enumerate(prompts)]
            try:
                for future in as_completed(futures):
                    row = future.result()
                    receipts.append(row)
                    for ctx, emitted, verifies in row["verify_rows"]:
                        totals.add({"al": (emitted, verifies)}, ctx_len=ctx)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        wall = time.perf_counter() - begin
        if Path(serving.prompts).read_bytes() != prompt_bytes:
            raise ValueError("prompt set changed during evaluation")
        if _identity(serving, info) != _identity(serving, server_info()):
            raise ValueError("serving model identity changed during evaluation")
        if serving_checkpoint_identity(spec.checkpoint) != checkpoint_identity:
            raise ValueError("serving checkpoint metadata changed during evaluation")
        result["checkpoint_identity"] = checkpoint_identity
        result["complete"] = True
    finally:
        result.update(request_count=len(receipts), **totals.result())
        result["sampling"] = {"temperature": serving.temperature, "top_p": serving.top_p,
                              "max_new_tokens": serving.max_new_tokens, "seed": spec.seed,
                              "concurrency": serving.concurrency, "protocol": serving.protocol,
                              "warmup_requests": serving.warmup_requests}
        if wall is not None:
            latencies = sorted(row["duration_seconds"] for row in receipts)
            output_tokens = sum(row["completion_tokens"] for row in receipts)
            result["performance"] = {"wall_seconds": wall, "completion_tokens": output_tokens,
                                     "output_tokens_per_second": output_tokens / wall,
                                     "request_latency_p50_seconds": statistics.median(latencies),
                                     "request_latency_p95_seconds": latencies[max(0, (95 * len(latencies) + 99) // 100 - 1)],
                                     "timing_scope": "nonstreaming HTTP end-to-end; excludes warmup"}
        if "al" in spec.metrics:
            result["definition"] = "committed output tokens / verify cycles, including actual correction/bonus tokens"
        atomic_json(attempt / "serving-results.json", result)
    return result
