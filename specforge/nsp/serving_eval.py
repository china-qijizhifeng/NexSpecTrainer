"""Request-scoped SGLang AL. Missing verifier telemetry is an error, never AL=1."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import requests

from specforge.nsp.metrics import MetricTotals, parse_bins
from specforge.nsp.store import atomic_json


def serving_terms(meta, completion_tokens, grouped):
    cycles = meta.get("spec_verify_cycles")
    if cycles is not None:
        if not isinstance(cycles, list) or not cycles:
            raise ValueError("spec_verify_cycles must contain observed verify cycles")
        seen = set()
        rows = []
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
            raise ValueError(
                "verify-cycle output total disagrees with completion_tokens"
            )
        return rows
    if grouped:
        raise ValueError(
            "dynamic ctx bins require spec_verify_cycles telemetry; request-level AL is insufficient"
        )
    verifies = meta.get("spec_verify_ct")
    if type(verifies) is not int or verifies <= 0:
        raise ValueError("missing positive spec_verify_ct; cannot measure AL")
    if type(completion_tokens) is not int or completion_tokens <= 0:
        raise ValueError("missing positive completion_tokens")
    return [(None, completion_tokens, verifies)]


def run_serving(spec, attempt):
    serving = spec.serving
    session = requests.Session()
    if serving.api_key_env:
        token = os.environ.get(serving.api_key_env)
        if not token:
            raise ValueError(
                f"missing credential environment variable {serving.api_key_env}"
            )
        session.headers["Authorization"] = f"Bearer {token}"
    url = serving.url.rstrip("/")
    info = session.get(url + serving.info_path, timeout=serving.timeout)
    info.raise_for_status()
    info = info.json()
    arguments = info.get("server_args", info)
    if arguments.get(serving.checkpoint_field) != spec.checkpoint:
        raise ValueError(
            "serving endpoint does not attest the requested draft checkpoint; use a dedicated matching server"
        )
    totals = MetricTotals(["al"], parse_bins(spec.ctx_bins) if spec.group_by else None)
    requests_count = 0
    try:
        with (
            Path(serving.prompts).open() as prompts,
            (Path(attempt) / "requests.jsonl").open("x") as receipts,
        ):
            for line in prompts:
                if not line.strip():
                    continue
                prompt = json.loads(line)
                if (
                    not isinstance(prompt.get("messages"), list)
                    or not prompt["messages"]
                ):
                    raise ValueError(
                        "each prompt must contain a nonempty messages array"
                    )
                request_id = "nsp-" + uuid.uuid4().hex
                response = session.post(
                    url + "/v1/chat/completions",
                    json={
                        "model": serving.model,
                        "messages": prompt["messages"],
                        "temperature": serving.temperature,
                        "top_p": serving.top_p,
                        "max_tokens": serving.max_new_tokens,
                        "stream": False,
                        "seed": spec.seed,
                    },
                    headers={"X-Request-Id": request_id},
                    timeout=serving.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                meta = payload["choices"][0].get(
                    "meta_info", payload.get("meta_info", {})
                )
                tokens = payload["usage"]["completion_tokens"]
                rows = serving_terms(meta, tokens, bool(spec.group_by))
                for ctx, emitted, verifies in rows:
                    totals.add({"al": (emitted, verifies)}, ctx_len=ctx)
                requests_count += 1
                receipts.write(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "sample_id": prompt.get("id"),
                            "completion_tokens": tokens,
                            "verify_rows": rows,
                        }
                    )
                    + "\n"
                )
                receipts.flush()
        if requests_count == 0:
            raise ValueError("prompt set is empty")
    except BaseException:
        atomic_json(
            Path(attempt) / "serving-results.json",
            {
                "complete": False,
                "completed_requests": requests_count,
                **totals.result(),
            },
        )
        raise
    final_info = session.get(url + serving.info_path, timeout=serving.timeout)
    final_info.raise_for_status()
    final_info = final_info.json()
    final_args = final_info.get("server_args", final_info)
    identity_fields = (
        serving.checkpoint_field,
        "model_path",
        "weight_version",
        "speculative_algorithm",
        "speculative_num_draft_tokens",
    )
    if any(arguments.get(key) != final_args.get(key) for key in identity_fields):
        raise ValueError("serving model identity changed during evaluation")
    result = {
        "runtime": {
            "version": info.get("version"),
            **{key: arguments.get(key) for key in identity_fields},
        },
        "schema_version": 1,
        "complete": True,
        "stage": "serving",
        "checkpoint": spec.checkpoint,
        "request_count": requests_count,
        "ctx_basis": "verify_prefix" if spec.group_by else None,
        "definition": "committed output tokens / verify cycles, including actual correction/bonus tokens",
        "sampling": {
            "temperature": serving.temperature,
            "top_p": serving.top_p,
            "max_new_tokens": serving.max_new_tokens,
            "seed": spec.seed,
            "concurrency": 1,
        },
        **totals.result(),
    }
    atomic_json(Path(attempt) / "serving-results.json", result)
    return result
