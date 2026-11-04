"""Exercise the full four-variant runner through actual local HTTP/SSE fixtures."""

import hashlib
import json
import math
import os
import stat
import struct
import tempfile
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import common
import evaluate
from observation_contract import (
    comparison,
    private_stream,
    validate_amendment,
    validate_resolved,
)


def ids_sha(ids):
    return hashlib.sha256(b"".join(struct.pack("<q", i) for i in ids)).hexdigest()


def main():
    state = {"group": None, "calls": Counter(), "budgets": Counter()}
    old = [{"id": f"old-{i}", "input_ids": [10 + i]} for i in range(2)]
    vals = [{"id": f"val-{i}", "input_ids": [30 + i]} for i in range(8)]
    tests = [
        {
            "id": f"test-{i}",
            "instance_id": f"instance-{i%36}",
            "input_ids": [1000 + i],
            "input_sha256": ids_sha([1000 + i]),
        }
        for i in range(247)
    ]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["sampling_params"]["temperature"] == 0
            assert body["sampling_params"]["ignore_eos"] is False
            budget = body["sampling_params"]["max_new_tokens"]
            group = state["group"]
            key = (group, tuple(body["input_ids"]))
            state["calls"][key] += 1
            state["budgets"][(group, budget)] += 1
            count = state["calls"][key]
            spec = group != "target-only"
            # Force A/A divergence and every speculative diagnostic/test to differ.
            token = 7 + (count % 2 if not spec else 3)
            ids = [token, 1] if budget > 1 else [token]
            meta = {
                "prompt_tokens": len(body["input_ids"]),
                "completion_tokens": len(ids),
                "finish_reason": (
                    {"type": "stop", "matched": 1}
                    if budget > 1
                    else {"type": "length", "length": 1}
                ),
                "cached_tokens": 1,
                "num_retractions": 0,
            }
            if spec:
                meta.update(
                    spec_verify_ct=1,
                    spec_num_proposed_drafts=5,
                    spec_num_correct_drafts=1,
                    spec_accept_length=len(ids),
                    spec_accept_rate=0.2,
                )
            if body.get("return_logprob"):
                meta["output_top_logprobs"] = [[[-0.1, token, None]]]
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/event-stream" if body["stream"] else "application/json",
            )
            self.end_headers()
            event = {"output_ids": ids, "meta_info": meta}
            if body["stream"]:
                self.wfile.write(
                    b"data: " + json.dumps(event).encode() + b"\n\ndata: [DONE]\n\n"
                )
            else:
                self.wfile.write(json.dumps(event).encode())
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "r3-eos-reference.json").write_text(
                json.dumps(
                    {
                        "input_sha256": ids_sha(vals[0]["input_ids"]),
                        "output_ids_through_eos": [99, 1],
                    }
                )
            )

            def launch(group, folder):
                state["group"] = group
                return object(), object()

            for mask in (0o077, 0o022):
                p = root / f"permissions-{mask}"
                p.mkdir()
                previous = os.umask(mask)
                try:
                    common.write(p / "public.json", {"passed": True})
                    with private_stream(p / "private.json") as stream:
                        stream.write("{}")
                    assert stat.S_IMODE((p / "public.json").stat().st_mode) == 0o644
                    assert stat.S_IMODE((p / "private.json").stat().st_mode) == 0o600
                    try:
                        private_stream(p / "private.json")
                    except FileExistsError:
                        pass
                    else:
                        raise AssertionError("private replay accepted")
                finally:
                    os.umask(previous)
            with (
                patch.object(evaluate, "ROOT", root),
                patch.object(common, "ROOT", root),
                patch.object(evaluate, "URL", f"http://127.0.0.1:{server.server_port}"),
                patch.object(evaluate, "launch", side_effect=launch),
                patch.object(evaluate, "stop"),
                patch("profile_decode.profile_validation"),
                patch("performance_contract.graph_receipt"),
                patch.object(evaluate, "verify_execution"),
                patch.object(
                    evaluate.shutil,
                    "disk_usage",
                    return_value=SimpleNamespace(free=100_000_000_000),
                ),
            ):
                result = evaluate.run_matrix(tests, vals, old)
            assert (
                result["execution_complete"]
                and result["final_request_audit"]["requests"] == 988
            )
            assert result["lossless_greedy_verified"] is False
            assert not (root / "target-only/aa-acceptance.json").exists()
            observation = json.loads(
                (root / "target-only/aa-observation.json").read_text()
            )
            assert (
                observation["all_equal"] is False
                and observation["historical_reference_matches"] is False
            )
            assert len(list(root.glob("*/test-*.json"))) == 988
            for group in evaluate.GROUPS:
                summary = json.loads((root / group / "summary.json").read_text())
                assert summary["requests"] == 247
                assert summary["cached_input_tokens"] == 247
                assert math.isfinite(summary["end_to_end_tokens_per_second"])
                assert len(summary["by_instance"]) == 36
                if group != "target-only":
                    assert summary["accept_rate"] == 0.2
                    assert summary["verify_weighted_accept_length"] == 2
                    assert summary["test_exact_match_requests_to_ar"] == 0
                    assert len(list((root / group).glob("prefix-probe-*.json"))) == 1
                    assert state["budgets"][(group, 1)] == 1
            for path in (
                list(root.glob("*/test-*.json"))
                + list(root.glob("*/diagnostic-*.json"))
                + list(root.glob("*/prefix-probe-*.json"))
            ):
                assert stat.S_IMODE(path.stat().st_mode) == 0o600
                assert (
                    stat.S_IMODE(path.with_suffix(".events.jsonl").stat().st_mode)
                    == 0o600
                )
            public = (root / "target-only/aa-observation.json").read_text()
            assert "output_ids" not in public and "output_top_logprobs" not in public
            # Identity mismatches and invalid accounting are still rejected.
            try:
                comparison(
                    {"id": "a", "input_sha256": "x"}, {"id": "b", "input_sha256": "x"}
                )
            except AssertionError:
                pass
            else:
                raise AssertionError("different request identities compared")
            row = {
                "output_ids": [7, 1],
                "meta_info": {
                    "spec_verify_ct": 1,
                    "spec_num_proposed_drafts": 4,
                    "spec_num_correct_drafts": 1,
                    "spec_accept_length": 2,
                    "spec_accept_rate": 0.25,
                },
            }
            try:
                evaluate.validate_stats(row, True)
            except AssertionError:
                pass
            else:
                raise AssertionError("corrupt proposed count accepted")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    protocol = json.loads(
        (Path(__file__).resolve().parent / "protocol.json").read_text()
    )
    validate_amendment(protocol)
    protocol["observational_amendment"]["lossless_claim_allowed"] = True
    try:
        validate_amendment(protocol)
    except AssertionError:
        pass
    else:
        raise AssertionError("lossless claim permitted")
    info = {
        "attention_backend": "dsv4",
        "moe_runner_backend": "flashinfer_mxfp4",
        "tp_size": 8,
        "ep_size": 8,
        "disable_radix_cache": False,
        "max_running_requests": 1,
        "chunked_prefill_size": -1,
        "cuda_graph_backend_prefill": "disabled",
        "cuda_graph_backend_decode": "full",
        "disable_overlap_schedule": False,
        "enable_deterministic_inference": False,
        "speculative_algorithm": "DSPARK",
        "speculative_dspark_block_size": 5,
        "api_key": "private-fixture",
    }
    assert "api_key" not in validate_resolved(info, "step25")
    print(
        json.dumps(
            {
                "passed": True,
                "real_http_sse_four_variant_requests": 988,
                "aa_and_historical_difference_continues": True,
                "no_false_acceptance_or_lossless_claim": True,
                "one_prefix_probe_per_spec_variant": True,
                "public0644_private0600": True,
                "strict_identity_and_counters_retained": True,
                "private_server_info_filtered": True,
                "scope": "CPU real HTTP/SSE runner matrix only; no GPU or model-quality result",
            }
        )
    )


if __name__ == "__main__":
    main()
