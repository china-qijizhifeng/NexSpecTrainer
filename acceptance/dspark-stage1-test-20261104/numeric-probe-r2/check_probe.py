"""CPU checks of actual HTTP/SSE capture, bounded logprob comparisons and resource guards."""

import argparse
import ast
import copy
import json
import logging
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import probe
from probe_contract import (
    SOURCE,
    body_for,
    compare,
    ids_sha,
    numeric_rows,
    request_plan,
    selected_input,
    validate_job,
    validate_resolved,
    verify_source,
)


def rejected(call):
    try:
        call()
    except (AssertionError, KeyError, ValueError):
        return
    raise AssertionError("invalid probe evidence accepted")


def fixture(ids):
    top = [[[-0.1 - i, 7 + i, None] for i in range(10)] for _ in ids]
    for token, row in zip(ids, top):
        if token == 1:
            row[0][1] = 1
    return {
        "output_ids": ids,
        "meta_info": {
            "completion_tokens": len(ids),
            "prompt_tokens": 4,
            "finish_reason": {"type": "stop", "matched": 1},
            "cached_tokens": 0,
            "num_retractions": 0,
            "output_top_logprobs": top,
            "output_token_logprobs": [[-0.1, token, None] for token in ids],
            "output_token_ids_logprobs": [
                [[-0.1, 7, None], [-1.1, 8, None]] for _ in ids
            ],
        },
    }


def check_native_override(source):
    path = source / "python/sglang/srt/arg_groups/overrides.py"
    tree = ast.parse(path.read_text())
    function = next(
        x
        for x in tree.body
        if isinstance(x, ast.FunctionDef)
        and x.name == "_enforce_disable_allreduce_fusion"
    )
    function.decorator_list = []
    namespace = {"logger": logging.getLogger("probe-cpu-check")}
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    invoke = namespace[function.name]
    assert (
        invoke(SimpleNamespace(enforce_disable_flashinfer_allreduce_fusion=False)) == {}
    )
    assert invoke(
        SimpleNamespace(enforce_disable_flashinfer_allreduce_fusion=True)
    ) == {"flashinfer_allreduce_fusion_backend": None}


def main(source):
    import yaml

    root = Path(__file__).resolve().parent
    job = yaml.safe_load((root / "eval-job.yaml").read_text())
    validate_job(job)
    pod = job["spec"]["tasks"][0]["template"]["spec"]
    bad = copy.deepcopy(job)
    bad["spec"]["tasks"][0]["template"]["spec"]["nodeName"] = "node054"
    rejected(lambda: validate_job(bad))
    for location in ("name", "JobAlias"):
        bad = copy.deepcopy(job)
        if location == "name":
            bad["metadata"]["name"] = "old-attempt"
        else:
            bad["metadata"]["annotations"]["JobAlias"] = "old-attempt"
        rejected(lambda: validate_job(bad))
    bad = copy.deepcopy(job)
    bad["spec"]["tasks"][0]["template"]["spec"]["containers"][0]["command"][
        1
    ] = "/gpfs/users/linlizhi/dspark-test-20261104-r7/run_eval.sh"
    rejected(lambda: validate_job(bad))
    assert pod["activeDeadlineSeconds"] == 14400
    assert set(m["name"] for m in pod["containers"][0]["volumeMounts"]) == {
        "gpfs",
        "shm",
    }
    info = {
        "attention_backend": "dsv4",
        "moe_runner_backend": "flashinfer_mxfp4",
        "tp_size": 8,
        "ep_size": 8,
        "disable_radix_cache": True,
        "max_running_requests": 1,
        "chunked_prefill_size": -1,
        "enable_deterministic_inference": False,
        "cuda_graph_backend_prefill": "disabled",
        "cuda_graph_backend_decode": "disabled",
        "enforce_disable_flashinfer_allreduce_fusion": False,
        "flashinfer_allreduce_fusion_backend": "auto",
        "api_key": "private-fixture-must-be-omitted",
    }
    assert "api_key" not in validate_resolved(info, "baseline")
    rejected(lambda: validate_resolved(info, "no-fused-allreduce"))
    info.update(
        enforce_disable_flashinfer_allreduce_fusion=True,
        flashinfer_allreduce_fusion_backend=None,
    )
    validate_resolved(info, "no-fused-allreduce")
    entry = {"input_ids": [4, 5, 6, 7], "budget": 32, "logprobs": True}
    body = body_for(entry, [7, 8])
    assert body["sampling_params"] == {
        "temperature": 0,
        "ignore_eos": False,
        "max_new_tokens": 32,
    }
    assert body["logprob_start_len"] == -1 and body["token_ids_logprob"] == [7, 8]
    result = fixture([7, 1])
    assert len(numeric_rows(result, [7, 8])) == 2
    corrupt = copy.deepcopy(result)
    corrupt["meta_info"]["output_top_logprobs"][0][0][0] = float("nan")
    rejected(lambda: numeric_rows(corrupt, [7, 8]))
    corrupt = copy.deepcopy(result)
    corrupt["meta_info"]["output_token_ids_logprobs"].pop()
    rejected(lambda: numeric_rows(corrupt, [7, 8]))
    changed = copy.deepcopy(result)
    changed["output_ids"][0] = 8
    changed["meta_info"]["output_token_ids_logprobs"][0][0][0] -= 0.01
    changed["meta_info"]["output_token_ids_logprobs"][1][0][0] -= 20
    delta = compare(result, changed)
    assert (
        delta["first_difference_zero_based"] == 0
        and delta["same_input_prefix_logprob_steps"] == 1
    )
    assert 0.009 < delta["max_abs_candidate_logprob_difference"] < 0.011
    pending, bodies = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            bodies.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for event in pending.pop(0):
                self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original_url = probe.URL
    probe.URL = f"http://127.0.0.1:{server.server_port}"
    try:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            pending.append(
                [
                    {"output_ids": [7], "meta_info": {"completion_tokens": 1}},
                    {"output_ids": [1], "meta_info": result["meta_info"]},
                ]
            )
            out = probe.capture(entry, [7, 8], folder / "good.json")
            assert out["output_ids"] == [7, 1]
            assert (folder / "good.json").stat().st_mode & 0o777 == 0o600
            assert (folder / "good.events.jsonl").stat().st_mode & 0o777 == 0o600
            assert bodies[-1] == body
            rejected(lambda: probe.capture(entry, [7, 8], folder / "good.json"))
            bad = fixture([7, 1, 9])
            pending.append([bad])
            rejected(lambda: probe.capture(entry, [7, 8], folder / "post-eos.json"))
            bad = fixture([7, 1])
            bad["meta_info"]["num_retractions"] = 1
            pending.append([bad])
            rejected(lambda: probe.capture(entry, [7, 8], folder / "retracted.json"))
    finally:
        probe.URL = original_url
        server.shutdown()
        server.server_close()
        thread.join()
    check_native_override(source)
    source_checked = False
    if SOURCE.exists():
        verify_source()
        row, prefix, candidates = selected_input()
        plan = request_plan(row, prefix)
        old = probe.old_prompts()
        assert len(plan) == 12 and sum(entry["logprobs"] for entry in plan) == 6
        for entry in plan:
            if entry["input_ids"] is None:
                entry["input_ids"] = old[entry["diagnostic"]]["input_ids"]
            if entry["kind"] == "history":
                assert ids_sha(entry["input_ids"]) == entry["recorded"]["input_sha256"]
            body_for(entry, candidates)
        assert [entry["budget"] for entry in plan] == [1024] * 6 + [32] * 4 + [1] * 2
        source_checked = True
    print(
        json.dumps(
            {
                "passed": True,
                "real_http_sse_capture": True,
                "finite_logprobs_and_exact_counts": True,
                "post_eos_rejected": True,
                "comparison_stops_at_prefix_divergence": True,
                "private_metadata_omitted": True,
                "native_disable_override_checked": True,
                "actual_private_validation_plan_checked": source_checked,
                "no_fixed_node": True,
                "scope": "CPU client/source/resource contracts; no GPU numeric inference",
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serving-source", type=Path, required=True)
    main(parser.parse_args().serving_source)
