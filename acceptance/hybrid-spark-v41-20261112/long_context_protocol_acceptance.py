"""Exercise the 64K harness through real detached nsp Runs and local HTTP.

This is a transport fixture. Token encoding and model output are synthetic;
official tokenizer/model acceptance is performed separately on YiCloud.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading

import long_context_acceptance as acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    model = folder / "fixture-checkpoint"
    model.mkdir()
    (model / "config.json").write_text("{}")
    def fake_encode(content, thinking):
        return [11] * (content.count(" reference") + len(content.replace(" reference", "").split()) + 9)
    prompts = acceptance.prepare_prompts(model, folder, encode=fake_encode)
    assert len(prompts) == 12
    assert sorted({len(row["input_ids"]) for row in prompts}) == [65535, 65536, 65537]
    seen = []
    current = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, payload):
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            assert self.path == "/get_server_info"
            self.respond(dict(server_args=current))

        def do_POST(self):
            assert self.path == "/generate"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert len(body["input_ids"]) in acceptance.INPUT_LENGTHS
            assert body["sampling_params"]["max_new_tokens"] == acceptance.MAX_NEW_TOKENS
            assert body["sampling_params"]["temperature"] == 0
            assert "sampling_seed" in body["sampling_params"] and "seed" not in body["sampling_params"]
            seen.append(dict(rid=body["rid"], input_tokens=len(body["input_ids"]),
                             sampling=body["sampling_params"]))
            self.respond(dict(output_ids=[13, 14, 15, 1], meta_info=dict(completion_tokens=4, spec_verify_ct=1)))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = [sys.executable, "-m", "specforge.nsp.cli", "--root", str(folder / "nsp")]
    run_receipts = {}
    try:
        for variant in ("target", "dspark", "hybrid"):
            current.clear()
            current.update(model_path=str(model), random_seed=42, context_length=acceptance.CONTEXT_LENGTH,
                chunked_prefill_size=acceptance.PREFILL_CHUNK, max_running_requests=1,
                speculative_algorithm=None if variant == "target" else "DSPARK",
                speculative_draft_model_path=None if variant == "target" else str(model),
                speculative_num_draft_tokens=16 if variant == "hybrid" else 6)
            destination = folder / variant
            destination.mkdir()
            status, result, attempt = acceptance.evaluate(base, model, variant, destination,
                folder / "measured-prompts.jsonl", server.server_port)
            rows = list(map(json.loads, (attempt / "requests.jsonl").read_text().splitlines()))
            assert len(rows) == result["stages"]["serving"]["request_count"] == 12
            assert {row["sample_id"]: row["input_ids"] for row in rows} == {row["id"]: row["input_ids"] for row in prompts}
            run_receipts[variant] = dict(id=status["id"], state=status["state"], measured_requests=len(rows))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert len(seen) == 39  # 12 measured + one separate warmup per variant.
    observed = [dict(sample_id=prompts[0]["id"], prefill_tokens=1, cycles=[
        dict(cycle_id=0, ctx_len=65536, committed_output_tokens=16),
        dict(cycle_id=1, ctx_len=65552, committed_output_tokens=4)])]
    summary = acceptance.summarize_cycles(observed, prompts)
    assert len(summary["long_commits_followed_by_decode"]) == 1
    observed[0]["cycles"][1]["ctx_len"] -= 1
    try:
        acceptance.summarize_cycles(observed, prompts)
    except ValueError:
        rejected = True
    else:
        raise AssertionError("corrupted 64K context advancement was accepted")
    receipt = dict(passed=True, scope=__doc__, nsp_runs=run_receipts, total_http_requests=len(seen),
        measured_per_variant=12, warmup_per_variant=1, input_lengths=list(acceptance.INPUT_LENGTHS),
        max_input_output_tokens=max(acceptance.INPUT_LENGTHS) + acceptance.MAX_NEW_TOKENS,
        server_context_length=acceptance.CONTEXT_LENGTH, corrupt_context_rejected=rejected)
    acceptance.write(folder / "summary.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
