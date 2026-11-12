"""Real detached CLI / HTTP / persistent-result integration using a protocol fixture.

The fixture verifies the transport contract, not SGLang model quality or speed.
"""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml


@pytest.mark.parametrize(
    ("protocol", "grouped", "http_status"),
    [("openai", False, 200), ("openai", True, 200),
     ("native", False, 200), ("native", True, 200), ("native", False, 400)],
)
def test_serving_run_persists_real_response_counts(tmp_path, protocol, grouped, http_status):
    checkpoint = tmp_path / "draft"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"protocol fixture, not loaded")
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(http_status if self.command == "POST" else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            assert self.path == "/get_server_info"
            self.respond({"speculative_draft_model_path": str(checkpoint)})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            if protocol == "native":
                assert self.path == "/generate"
                assert body["sampling_params"] == {
                    "temperature": 0.0, "top_p": 1.0,
                    "max_new_tokens": 128, "sampling_seed": 42,
                }
            else:
                assert self.path == "/v1/chat/completions"
                assert body["seed"] == 42
            if http_status == 400:
                self.respond({"error": {"message": "invalid sampling field", "detail": "x" * 10_000}})
            elif protocol == "native":
                self.respond({"output_ids": list(range(7)),
                              "meta_info": {"spec_verify_ct": 2, "completion_tokens": 7}})
            else:
                self.respond({"choices": [{"meta_info": {"spec_verify_ct": 2}}],
                              "usage": {"completion_tokens": 7}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    prompt = tmp_path / "prompts.jsonl"
    prompt.write_text(json.dumps({"id": "p1", "input_ids": [10, 11],
                                 "messages": [{"role": "user", "content": "hi"}]}) + "\n")
    spec = {
        "metrics": ["al"],
        "serving": {
            "url": f"http://127.0.0.1:{server.server_port}",
            "prompts": str(prompt),
            "model": "fixture",
            "protocol": protocol,
        },
    }
    if grouped:
        spec["group_by"] = "ctx-len"
    config = tmp_path / "eval.yaml"
    config.write_text(yaml.safe_dump(spec))
    base = [sys.executable, "-m", "specforge.nsp.cli", "--root", str(tmp_path / "runs")]
    try:
        submitted = subprocess.run(
            base + ["eval", str(checkpoint), "-c", str(config), "--json"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        run_id = json.loads(submitted.stdout)["id"]
        completed = subprocess.run(
            base
            + [
                "watch",
                run_id,
                "--until",
                "terminal",
                "--timeout",
                "20",
                "--interval",
                "0.1",
                "--jsonl",
            ],
            capture_output=True,
            text=True,
            timeout=25,
        )
        failed = grouped or http_status == 400
        assert completed.returncode == (1 if failed else 0), (
            completed.stdout + completed.stderr
        )
        results = subprocess.run(
            base + ["results", run_id, "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(results.stdout)
        assert payload["complete"] is not failed
        if http_status == 400:
            assert "serving HTTP 400" in payload["error"]
            assert "invalid sampling field" in payload["error"]
            assert len(payload["error"]) < 4300
            assert payload["stages"]["serving"]["complete"] is False
        elif grouped:
            assert "spec_verify_cycles" in payload["error"]
            assert payload["stages"]["serving"]["complete"] is False
        else:
            assert payload["stages"]["serving"]["overall"]["al"] == {
                "value": 3.5,
                "numerator": 7.0,
                "denominator": 2.0,
            }
        assert len(seen) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
