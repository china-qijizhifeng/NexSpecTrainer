"""CPU boundary regressions for the deployment acceptance and Service contract."""

import copy
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from accept_worker import Client, native_audit
from platform_ops import own_services


def rejected(fn):
    try:
        fn()
    except AssertionError:
        return
    raise AssertionError("invalid boundary was accepted")


def main():
    valid = {
        "output_ids": [20, 21, 1],
        "meta_info": {
            "completion_tokens": 3,
            "finish_reason": {"type": "stop", "matched": 1},
            "spec_verify_ct": 1,
            "spec_num_proposed_drafts": 5,
            "spec_num_correct_drafts": 4,
            "spec_accept_length": 3.0,
            "spec_accept_rate": 0.8,
        },
    }
    # Native accepted drafts may include work beyond the terminal EOS.
    assert native_audit(valid, 32)["raw_correct"] == 4
    one = {
        "output_ids": [1],
        "meta_info": {
            "completion_tokens": 1,
            "finish_reason": {"type": "stop", "matched": 1},
        },
    }
    assert native_audit(one, 32)["verify"] == 0
    for change in ("after-eos", "count", "proposed", "rate", "nan"):
        bad = copy.deepcopy(valid)
        if change == "after-eos":
            bad["output_ids"] = [20, 1, 21]
        elif change == "count":
            bad["meta_info"]["completion_tokens"] = 4
        elif change == "proposed":
            bad["meta_info"]["spec_num_proposed_drafts"] = 10
        elif change == "rate":
            bad["meta_info"]["spec_accept_rate"] = 0.4
        else:
            bad["meta_info"]["spec_accept_length"] = float("nan")
        rejected(lambda: native_audit(bad, 32))
    row = {"creator": "linlizhi", "namespace": "qiji"}
    data = {"code": 0, "data": {"total": 1, "services": [row]}}
    assert own_services(lambda path: data) == data
    rejected(
        lambda: own_services(
            lambda path: {"code": 0, "data": {"total": 2, "services": [row]}}
        )
    )
    rejected(
        lambda: own_services(
            lambda path: {
                "code": 0,
                "data": {"total": 1, "services": [{**row, "creator": "someone-else"}]},
            }
        )
    )

    class Fixture(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.headers["Authorization"] == "Bearer fixture-private-key"
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b": comment\r\n\r\n")
            self.wfile.write(b"data: " + json.dumps(valid).encode() + b"\r\n\r\n")
            if self.path != "/truncated":
                self.wfile.write(b"data: [DONE]\r\n\r\n")

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as directory:
            client = Client(
                f"http://127.0.0.1:{server.server_port}",
                "fixture-private-key",
                Path(directory),
            )
            result = client.call("sse", "/generate", {}, True)
            assert result["done"] and len(result["events"]) == 1
            native_audit(result["events"][0]["response"], 32)
            rejected(lambda: client.call("truncated", "/truncated", {}, True))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    print(
        json.dumps(
            {
                "passed": True,
                "scope": "CPU native EOS/count/terminal-block, Service pagination/ownership, actual HTTP SSE completion; no GPU or numerical equivalence claim",
            }
        )
    )


if __name__ == "__main__":
    main()
