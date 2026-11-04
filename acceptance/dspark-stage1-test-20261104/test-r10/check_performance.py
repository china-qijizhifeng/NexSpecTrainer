"""Exercise cache guards, real plain-text profiler HTTP, and private trace summaries."""

import gzip
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import evaluate
from observation_contract import validate_resolved
from performance_contract import graph_receipt, validate_cache_counts
from profile_decode import profile_validation


def rejected(call):
    try:
        call()
    except AssertionError:
        return
    raise AssertionError("invalid performance contract accepted")


def main():
    for cached in (0, 1, 10):
        validate_cache_counts({"cached_tokens": cached, "prompt_tokens": 10}, 10)
    for cached in (-1, 11, 1.5, True):
        rejected(
            lambda: validate_cache_counts(
                {"cached_tokens": cached, "prompt_tokens": 10}, 10
            )
        )
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
        "speculative_algorithm": None,
    }
    validate_resolved(info, "target-only")
    rejected(
        lambda: validate_resolved(
            {**info, "cuda_graph_backend_decode": "disabled"}, "target-only"
        )
    )
    rejected(
        lambda: validate_resolved({**info, "disable_radix_cache": True}, "target-only")
    )
    state = {"calls": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["calls"].append(self.path)
            if self.path == "/start_profile":
                assert body["num_steps"] == 8
                assert body["record_shapes"] is False
                target = Path(body["output_dir"])
                for rank in range(8):
                    with gzip.open(
                        target / f"rank-{rank}.trace.json.gz", "wt"
                    ) as stream:
                        json.dump(
                            {
                                "traceEvents": [
                                    {
                                        "ph": "X",
                                        "cat": "kernel",
                                        "name": "gather",
                                        "dur": 10,
                                        "args": {"private": "omitted"},
                                    }
                                ]
                            },
                            stream,
                        )
                text = b"Start profiling.\n"
            else:
                assert self.path == "/stop_profile"
                text = b"Stop profiling. This will take some time.\n"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(text)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)

            def generate(row, length, path, on_first_token):
                assert length == 64
                on_first_token()
                return {"input_sha256": "fixture", "output_ids": [7, 1]}

            with patch.object(
                evaluate, "URL", f"http://127.0.0.1:{server.server_port}"
            ):
                profile_validation(folder, {}, generate, evaluate.request)
            summary = json.loads((folder / "decode-profile-summary.json").read_text())
            assert summary["trace_count"] == 8 and summary["validation_only"]
            assert "private" not in (folder / "decode-profile-summary.json").read_text()
            assert state["calls"] == ["/start_profile"]
            (folder / "server.log").write_text("Decode batch, cuda graph: False\n")
            rejected(lambda: graph_receipt(folder))
            (folder / "server.log").write_text("Decode batch, cuda graph: True\n")
            graph_receipt(folder)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    print(
        json.dumps(
            {
                "passed": True,
                "prefix_count_guards": True,
                "actual_graph_required": True,
                "real_plain_text_profile_http": True,
                "bounded_private_traces": True,
            }
        )
    )


if __name__ == "__main__":
    main()
