"""Single TP8/EP8 worker, four fixed variants, raw SSE and request counters."""

import collections
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from common import ROOT, TARGET, phase, sha, verify_execution, write

URL = "http://127.0.0.1:30000"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
GROUPS = ("target-only", "official-mtp", "step25", "step118")
EOS_IDS = {1}


def request(path, body=None, timeout=30):
    req = urllib.request.Request(
        URL + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with OPENER.open(req, timeout=timeout) as f:
        return json.load(f)


def sse_events(lines):
    data = []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                payload = "\n".join(data)
                data = []
                if payload == "[DONE]":
                    return
                yield json.loads(payload)
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        payload = "\n".join(data)
        if payload != "[DONE]":
            yield json.loads(payload)


def merge_ids(previous, event):
    ids = event.get("output_ids", [])
    total = event["meta_info"]["completion_tokens"]
    if len(ids) == total:
        assert ids[: len(previous)] == previous, "non-prefix cumulative stream"
        return list(ids)
    assert len(previous) + len(ids) == total, "inconsistent delta stream count"
    return previous + ids


def generate(row, length, path, stream=True, logprob=False):
    assert (
        shutil.disk_usage(ROOT).free > 50_000_000_000
    ), "GPFS below temporary 50GB write reserve"
    body = {
        "input_ids": row["input_ids"],
        "stream": stream,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": length,
            "ignore_eos": False,
        },
    }
    if logprob:
        body.update(return_logprob=True, top_logprobs_num=10)
    begin = time.monotonic()
    first = None
    ids = []
    last = None
    events = []
    with path.with_suffix(".events.jsonl").open("x") as log:
        if stream:
            req = urllib.request.Request(
                URL + "/generate",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with OPENER.open(req, timeout=2400) as response:
                for event in sse_events(response):
                    elapsed = time.monotonic() - begin
                    if "error" in event:
                        raise RuntimeError(str(event))
                    ids = merge_ids(ids, event)
                    if ids and first is None:
                        first = elapsed
                    last = event
                    log.write(
                        json.dumps({"elapsed_seconds": elapsed, "response": event})
                        + "\n"
                    )
                    log.flush()
                    events.append(
                        {"elapsed_seconds": elapsed, "completion_tokens": len(ids)}
                    )
        else:
            last = request("/generate", body, 2400)
            ids = last["output_ids"]
            first = None
            log.write(
                json.dumps(
                    {"elapsed_seconds": time.monotonic() - begin, "response": last}
                )
                + "\n"
            )
    duration = time.monotonic() - begin
    assert last, "empty generation response"
    meta = last["meta_info"]
    validate_completion(ids, meta, length, len(row["input_ids"]))
    count = len(ids)
    first_count = next(
        (
            event["completion_tokens"]
            for event in events
            if event["completion_tokens"] > 0
        ),
        count,
    )
    assert math.isfinite(duration) and duration > 0
    assert first is None or 0 <= first <= duration
    input_sha = hashlib.sha256(
        b"".join(struct.pack("<q", i) for i in row["input_ids"])
    ).hexdigest()
    out = {
        "id": row["id"],
        "instance_id": row.get("instance_id"),
        "input_sha256": input_sha,
        "input_tokens": len(row["input_ids"]),
        "output_ids": ids,
        "meta_info": meta,
        "duration_seconds": duration,
        "ttft_seconds": first,
        "events": events,
        "output_tokens_per_second": count / duration,
        "requested_max_new_tokens": length,
        "ignore_eos": False,
        "first_chunk_tokens": first_count,
        "tpot_seconds": (
            None
            if first is None or count == first_count
            else (duration - first) / (count - first_count)
        ),
        "stream_timing_scope": "chunk arrival timestamps; no inferred token-level times",
    }
    write(path, out)
    return out


def validate_completion(ids, meta, budget, input_tokens):
    assert 1 <= len(ids) <= budget, (len(ids), budget)
    assert all(isinstance(token, int) and token >= 0 for token in ids)
    assert (
        meta["completion_tokens"] == len(ids) and meta["prompt_tokens"] == input_tokens
    )
    finish = meta["finish_reason"]
    first_eos = next((i for i, token in enumerate(ids) if token in EOS_IDS), None)
    if first_eos is not None:
        assert first_eos == len(ids) - 1, "tokens emitted after first EOS"
        assert finish["type"] == "stop" and finish["matched"] in EOS_IDS, finish
    else:
        assert (
            len(ids) == budget
            and finish["type"] == "length"
            and finish["length"] == budget
        ), finish


def prefix(a, b):
    return next(
        (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
    )


def validate_stats(out, spec):
    m = out["meta_info"]
    verifies = m.get("spec_verify_ct", 0)
    assert isinstance(verifies, int) and verifies >= 0
    if spec and verifies:
        assert m["spec_num_proposed_drafts"] == verifies * 5
        assert 0 <= m["spec_num_correct_drafts"] <= m["spec_num_proposed_drafts"]
        assert math.isfinite(m["spec_accept_length"]) and math.isfinite(
            m["spec_accept_rate"]
        )
        assert math.isclose(
            m["spec_accept_length"], len(out["output_ids"]) / verifies, rel_tol=1e-6
        )
        assert math.isclose(
            m["spec_accept_rate"],
            m["spec_num_correct_drafts"] / m["spec_num_proposed_drafts"],
            rel_tol=1e-6,
        )
    else:
        assert verifies == 0
        assert not m.get("spec_num_correct_drafts", 0) and not m.get(
            "spec_num_proposed_drafts", 0
        )
        if spec:
            assert (
                len(out["output_ids"]) == 1
            ), "multiple output tokens without a verify step"


def aggregate(outputs, spec):
    count = len(outputs)
    tokens = sum(len(x["output_ids"]) for x in outputs)
    elapsed = sum(x["duration_seconds"] for x in outputs)
    result = {
        "requests": count,
        "output_tokens": tokens,
        "request_seconds": elapsed,
        "end_to_end_tokens_per_second": tokens / elapsed,
        "mean_ttft_seconds": statistics.mean(x["ttft_seconds"] for x in outputs),
        "mean_tpot_seconds": (
            statistics.mean(valid_tpot)
            if (
                valid_tpot := [
                    x["tpot_seconds"] for x in outputs if x["tpot_seconds"] is not None
                ]
            )
            else None
        ),
        "requests_with_defined_tpot": len(valid_tpot),
        "finish_counts": dict(
            collections.Counter(
                x["meta_info"]["finish_reason"]["type"] for x in outputs
            )
        ),
    }
    if spec:
        verifies = sum(x["meta_info"].get("spec_verify_ct", 0) for x in outputs)
        correct = sum(x["meta_info"].get("spec_num_correct_drafts", 0) for x in outputs)
        proposed = sum(
            x["meta_info"].get("spec_num_proposed_drafts", 0) for x in outputs
        )
        result.update(
            verify_ct=verifies,
            correct_drafts=correct,
            proposed_drafts=proposed,
            verify_weighted_accept_length=tokens / verifies if verifies else None,
            accept_rate=correct / proposed if proposed else None,
            counter_scope="Raw native verify/draft counters include the terminal block; emitted tokens stop at EOS or budget. No equality between accepted drafts and emitted tokens is assumed.",
        )
    return result


def launch(group, folder):
    model = (
        TARGET if group in ("target-only", "official-mtp") else ROOT / f"export-{group}"
    )
    cmd = [
        "sglang",
        "serve",
        "--model-path",
        str(model),
        "--trust-remote-code",
        "--tp-size",
        "8",
        "--ep-size",
        "8",
        "--attention-backend",
        "dsv4",
        "--moe-runner-backend",
        "flashinfer_mxfp4",
        "--engram-config",
        '{"placement":"cpu"}',
        "--mem-fraction-static",
        "0.8",
        "--context-length",
        "69632",
        "--max-total-tokens",
        "131072",
        "--swa-full-tokens-ratio",
        "0.8",
        "--max-prefill-tokens",
        "65536",
        "--max-running-requests",
        "1",
        "--skip-tokenizer-init",
        "--chunked-prefill-size",
        "-1",
        "--disable-radix-cache",
        "--no-enable-decoder-swa-bounded-replay",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-backend-decode",
        "disabled",
        "--host",
        "127.0.0.1",
        "--port",
        "30000",
    ]
    if group != "target-only":
        cmd += [
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-dspark-block-size",
            "5",
        ]
    env = dict(os.environ)
    assert not any(
        "SIMULATE" in k and v not in ("", "0", "false", "False") for k, v in env.items()
    ), "simulation environment present"
    env["SGLANG_RAGGED_VERIFY_MODE"] = "static"
    write(
        folder / "launch.json",
        {
            "command": cmd,
            "environment": {
                k: v
                for k, v in env.items()
                if k.startswith(
                    (
                        "SGLANG_",
                        "DG_",
                        "TRITON_",
                        "CUDA_CACHE",
                        "TILELANG_",
                        "FLASHINFER_",
                    )
                )
            },
        },
    )
    log = (folder / "server.log").open("x")
    p = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
    )
    write(
        folder / "server-process.json",
        {
            "pid": p.pid,
            "pgid": p.pid,
            "start_ticks": Path(f"/proc/{p.pid}/stat")
            .read_text()
            .rsplit(")", 1)[1]
            .split()[19],
        },
    )
    begin = time.monotonic()
    try:
        while time.monotonic() - begin < 5400:
            if p.poll() is not None:
                raise RuntimeError(f"{group} server exited {p.returncode}")
            try:
                # A healthy no-tokenizer server supports /health without generating tokens.
                with OPENER.open(URL + "/health", timeout=3) as response:
                    if response.status == 200:
                        info = request("/get_server_info")
                        write(folder / "server-info.json", info)
                        write(
                            folder / "readiness.json",
                            {"seconds": time.monotonic() - begin},
                        )
                        return p, log
            except (OSError, ValueError):
                pass
            time.sleep(5)
        raise TimeoutError("bounded readiness wait expired")
    except BaseException:
        stop(p, log)
        raise


def stop(p, log):
    # Only this worker's newly created process group, never name/prefix matching.
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        p.wait(timeout=90)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait(timeout=30)
    log.close()


def main():
    verify_execution()
    from common import verify_export

    for step in (25, 118):
        verify_export(ROOT, step, read_payload=True)
    imported = json.loads((ROOT / "image-import/result.json").read_text())
    from node_contract import validate_node

    validate_node(os.environ["NODE_NAME"])
    assert imported["passed"] and imported["node"] == os.environ["NODE_NAME"]
    assert len(imported["gpus"]) == 8 and all("H100" in row for row in imported["gpus"])
    assert (
        imported["manifest_digest"]
        == "sha256:c6da252c6837bb062184b01c7aa2df7a2aadb1ed4944f236d7027174c73c45fa"
    )
    assert json.loads((ROOT / "export-result.json").read_text())["passed"]
    import torch
    from google.protobuf.internal.api_implementation import Type

    assert torch.cuda.device_count() == 8 and all(
        torch.cuda.get_device_capability(i) == (9, 0)
        and "H100" in torch.cuda.get_device_name(i)
        for i in range(8)
    )
    versions = {
        n: importlib.metadata.version(n)
        for n in ("sglang", "torch", "transformers", "protobuf", "flashinfer-python")
    }
    assert "38d9e4a3f" in versions["sglang"], versions
    base = Path(importlib.util.find_spec("sglang").origin).parent
    source_pins = json.loads((ROOT / "serving-source-pins.json").read_text())
    for name, digest in source_pins.items():
        assert sha(base / name) == digest, name
    worker_cpu_checks = []
    for script, extra in (
        ("check_runner.py", []),
        ("check_eos.py", ["--serving-source", str(base)]),
    ):
        result = json.loads(
            subprocess.check_output(
                [sys.executable, str(ROOT / script), *extra], text=True
            )
        )
        assert result["passed"]
        worker_cpu_checks.append({"script": script, **result})
    write(
        ROOT / "worker-cpu-regression.json",
        {"passed": True, "checks": worker_cpu_checks},
    )
    write(
        ROOT / "eval-environment.json",
        {
            "versions": versions,
            "python": sys.version,
            "protobuf_backend": Type(),
            "hostname": platform.node(),
            "node_name": os.environ.get("NODE_NAME"),
            "gpus": [torch.cuda.get_device_name(i) for i in range(8)],
            "source_files_sha256": source_pins,
            "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
        },
    )
    assert (
        set([json.loads((TARGET / "config.json").read_text())["eos_token_id"]])
        == EOS_IDS
    )
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert (
        protocol["generation"]["ignore_eos"] is False
        and protocol["generation"]["max_new_tokens"] == 1024
    )
    tests = [
        json.loads(x) for x in (ROOT / "test-inputs.jsonl").read_text().splitlines()
    ]
    vals = [
        json.loads(x)
        for x in (ROOT / "validation-inputs.jsonl").read_text().splitlines()
    ]
    old = [
        {
            "id": "old-prompt-0",
            "input_ids": [
                0,
                128803,
                65106,
                1192,
                270,
                16277,
                11167,
                4181,
                69,
                67411,
                90477,
                38774,
                16,
                128804,
                128822,
            ],
        },
        {
            "id": "old-prompt-1",
            "input_ids": [
                0,
                128803,
                65106,
                3939,
                25627,
                4216,
                56113,
                11096,
                2231,
                5434,
                5854,
                18470,
                16,
                128804,
                128822,
            ],
        },
    ]
    assert (
        len(tests) == 247
        and len({row["instance_id"] for row in tests}) == 36
        and len(vals) == 8
    )
    diagnostics = old + vals
    comparisons = {}
    all_results = {}
    for group in GROUPS:
        spec = group != "target-only"
        folder = ROOT / group
        folder.mkdir()
        phase("server_starting", group=group)
        p, log = launch(group, folder)
        try:
            phase("validation_diagnostics", group=group)
            # Warm-up is separate, EOS-aware and capped32 for each variant.
            generate(old[1], 32, folder / "warmup.json")
            diag = []
            for i, row in enumerate(diagnostics):
                n = 1024
                result = generate(row, n, folder / f"diagnostic-{i}.json")
                validate_stats(result, spec)
                diag.append(result)
                if not spec:
                    repeat = generate(row, n, folder / f"diagnostic-{i}-repeat.json")
                    validate_stats(repeat, False)
                    if (
                        result["output_ids"] != repeat["output_ids"]
                        or result["meta_info"]["finish_reason"]
                        != repeat["meta_info"]["finish_reason"]
                    ):
                        write(
                            folder / "correctness-failure.json",
                            {"reason": "target-only A/A not stable", "id": row["id"]},
                        )
                        raise RuntimeError("target-only A/A mismatch; stop before test")
                    if row["id"] == vals[0]["id"]:
                        historical = json.loads(
                            (ROOT / "r3-eos-reference.json").read_text()
                        )
                        assert result["input_sha256"] == historical["input_sha256"]
                        assert (
                            result["output_ids"] == historical["output_ids_through_eos"]
                        ), "original r3 EOS-prefix repro changed"
                        write(
                            folder / "r3-eos-repro.json",
                            {
                                "passed": True,
                                "input_sha256": result["input_sha256"],
                                "tokens": len(result["output_ids"]),
                                "finish_reason": result["meta_info"]["finish_reason"],
                            },
                        )
            if spec:
                baseline = comparisons["target-only"]
                checks = []
                for i, (row, result, ref) in enumerate(
                    zip(diagnostics, diag, baseline)
                ):
                    k = prefix(result["output_ids"], ref["output_ids"])
                    check = {
                        "id": row["id"],
                        "match_tokens": k,
                        "compared_tokens": len(ref["output_ids"]),
                        "equal": result["output_ids"] == ref["output_ids"],
                    }
                    if not check["equal"] and k < min(
                        len(ref["output_ids"]), len(result["output_ids"])
                    ):
                        # One pre-declared common-prefix prefill probe, no iterative retries/tuning.
                        probe = {
                            **row,
                            "input_ids": row["input_ids"] + ref["output_ids"][:k],
                        }
                        scored = generate(
                            probe,
                            1,
                            folder / f"prefix-probe-{i}.json",
                            stream=False,
                            logprob=True,
                        )
                        check.update(
                            target_ar_token=ref["output_ids"][k],
                            spec_token=result["output_ids"][k],
                            fresh_prefill_token=scored["output_ids"][0],
                            fresh_prefill_top_logprobs=scored["meta_info"].get(
                                "output_top_logprobs"
                            ),
                        )
                    checks.append(check)
                write(
                    folder / "greedy-comparison.json",
                    {
                        "all_equal": all(x["equal"] for x in checks),
                        "samples": checks,
                        "scope": "EOS-aware diagnostic matrix; known unresolved FP4 AR/verify numerical-path discrepancy; no lossless claim if any mismatch",
                    },
                )
            if not spec:
                write(
                    folder / "aa-acceptance.json",
                    {
                        "passed": True,
                        "diagnostics": len(diag),
                        "max_new_tokens": 1024,
                        "ignore_eos": False,
                        "complete_outputs_and_finish_reasons_match": True,
                    },
                )
            comparisons[group] = diag
            phase("test_running", group=group, completed=0, total=len(tests))
            outputs = []
            for i, row in enumerate(tests):
                result = generate(row, 1024, folder / f"test-{i:04d}.json")
                validate_stats(result, spec)
                outputs.append(result)
                phase(
                    "test_running",
                    group=group,
                    completed=i + 1,
                    total=len(tests),
                    last_id=row["id"],
                )
            summary = aggregate(outputs, spec)
            summary["by_instance"] = {
                key: aggregate([x for x in outputs if x["instance_id"] == key], spec)
                for key in sorted({x["instance_id"] for x in outputs})
            }
            if spec:
                summary["greedy_diagnostics_all_equal"] = json.loads(
                    (folder / "greedy-comparison.json").read_text()
                )["all_equal"]
                base_results = all_results["target-only"]
                exact = sum(
                    x["output_ids"] == b["output_ids"]
                    for x, b in zip(outputs, base_results)
                )
                summary["test_exact_match_requests_to_ar"] = exact
                summary["test_mean_matching_prefix_tokens_to_ar"] = statistics.mean(
                    prefix(x["output_ids"], b["output_ids"])
                    for x, b in zip(outputs, base_results)
                )
            write(folder / "summary.json", summary)
            all_results[group] = outputs
            phase(
                "variant_complete",
                group=group,
                summary={k: v for k, v in summary.items() if k != "by_instance"},
            )
        finally:
            stop(p, log)
    summaries = {g: json.loads((ROOT / g / "summary.json").read_text()) for g in GROUPS}
    out = {
        "execution_complete": True,
        "test_trajectories": 247,
        "test_instances": 36,
        "variants": summaries,
        "lossless_greedy_verified": all(
            summaries[g].get("greedy_diagnostics_all_equal", False)
            and summaries[g]["test_exact_match_requests_to_ar"] == 247
            for g in GROUPS[1:]
        ),
        "generation": {"ignore_eos": False, "max_new_tokens": 1024},
        "scope": "EOS-terminated greedy held-out complete-prefix replay capped1024; no SWE solve-rate or production-concurrency claim",
    }
    for group, outputs in all_results.items():
        assert len(outputs) == 247
        for row, result in zip(tests, outputs):
            assert (
                row["id"] == result["id"]
                and row["input_sha256"] == result["input_sha256"]
            )
            validate_completion(
                result["output_ids"], result["meta_info"], 1024, len(row["input_ids"])
            )
            validate_stats(result, group != "target-only")
            for key in ("duration_seconds", "output_tokens_per_second", "ttft_seconds"):
                assert math.isfinite(result[key]) and result[key] >= 0, (
                    group,
                    row["id"],
                    key,
                )
            assert (
                result["tpot_seconds"] is None
                or math.isfinite(result["tpot_seconds"])
                and result["tpot_seconds"] >= 0
            )
    verify_execution()
    out["final_request_audit"] = {
        "passed": True,
        "requests": 247 * 4,
        "input_sha256_matched": True,
        "eos_and_finite_metrics_verified": True,
    }
    write(ROOT / "result.json", out)
    phase(
        "evaluation_complete", lossless_greedy_verified=out["lossless_greedy_verified"]
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write(
            ROOT / "eval-failure.json",
            {"type": type(exc).__name__, "reason": str(exc), "automatic_retry": False},
        )
        phase("evaluation_failed", error_type=type(exc).__name__, reason=str(exc))
        raise
