"""Real HTTP and detached nsp lifecycle gate using deterministic protocol fixtures.

This exercises transport, concurrent receipts and calibration, not model quality.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def command(base, argv, folder, name, *, expected=0):
    result = subprocess.run(base + argv, capture_output=True, text=True, timeout=90)
    (folder / (name + ".stdout")).write_text(result.stdout)
    (folder / (name + ".stderr")).write_text(result.stderr)
    assert result.returncode == expected, (argv, result.returncode, result.stdout, result.stderr)
    return result.stdout


def watch(base, run_id, folder, name, *, succeeds=True):
    command(base, ["watch", run_id, "--until", "terminal", "--timeout", "60",
                  "--interval", "0.1", "--jsonl"], folder, name + "-watch",
            expected=0 if succeeds else 1)
    status = json.loads(command(base, ["show", run_id, "--json"], folder, name + "-show"))
    assert status["state"] == ("succeeded" if succeeds else "failed"), status
    return json.loads(command(base, ["results", run_id, "--json"], folder, name + "-results"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    checkpoint = folder / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    (checkpoint / "model.safetensors").write_bytes(b"protocol fixture only; never loaded")
    lock = threading.Lock()
    state = dict(active=0, peak=0, gets=0, requests=[], failure=None)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, payload):
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            assert self.path == "/get_server_info"
            with lock:
                state["gets"] += 1
            self.send(dict(version="protocol-fixture", server_args={
                "model_path": str(checkpoint), "speculative_draft_model_path": str(checkpoint),
                "speculative_algorithm": "DSPARK", "hybrid_config": "fixture-v1"}))

        def do_POST(self):
            assert self.path == "/generate"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["rid"] == self.headers["X-Request-Id"]
            assert body["stream"] is False
            assert body["sampling_params"]["temperature"] == 0
            marker = body["input_ids"][0] - 1000
            assert set(body["sampling_params"]) == {
                "temperature", "top_p", "max_new_tokens", "sampling_seed"}
            assert body["sampling_params"]["sampling_seed"] == 42 + marker
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                state["requests"].append(body)
            try:
                time.sleep(0.04)
                output = [200 + marker * 20 + k for k in range(16)]
                cycles = [dict(cycle_id=0, ctx_len=len(body["input_ids"]), committed_output_tokens=8),
                          dict(cycle_id=1, ctx_len=len(body["input_ids"]) + 8, committed_output_tokens=8)]
                meta = dict(completion_tokens=len(output), prompt_tokens=len(body["input_ids"]),
                            spec_verify_ct=2, spec_verify_cycles=cycles)
                if state["failure"] == "missing-cycles":
                    meta.pop("spec_verify_cycles")
                elif state["failure"] == "malformed-output":
                    output.pop()
                self.send(dict(output_ids=output, meta_info=meta))
            finally:
                with lock:
                    state["active"] -= 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = [sys.executable, "-m", "specforge.nsp.cli", "--root", str(folder / "nsp")]
    prompts = folder / "prompts.jsonl"
    rows = [dict(id=f"trace-{i}", input_ids=[1000 + i] + list(range(200 + i * 20, 216 + i * 20)) * 3)
            for i in range(8)]
    prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    spec = dict(metrics=["al", "performance"], group_by="ctx-len", ctx_bins=[0, 50, 60, "inf"],
        serving=dict(protocol="native", url=f"http://127.0.0.1:{server.server_port}",
            model="fixture", prompts=str(prompts), max_new_tokens=16, concurrency=4, warmup_requests=2,
            expected_server_args=dict(speculative_algorithm="DSPARK", hybrid_config="fixture-v1")))
    config = folder / "eval.json"
    write(config, spec)
    report = dict(scope="real HTTP/detached nsp protocol fixtures", gpu_tested=False)
    try:
        command(base, ["eval", str(checkpoint), "-c", str(config), "--plan"], folder, "eval-plan")
        submit = ["eval", str(checkpoint), "-c", str(config), "--request-id", "native-fixture", "--json"]
        run_id = json.loads(command(base, submit, folder, "eval-submit"))["id"]
        result = watch(base, run_id, folder, "eval")
        serving = result["stages"]["serving"]
        assert serving["request_count"] == 8
        assert serving["overall"]["al"] == dict(value=8.0, numerator=128.0, denominator=16.0)
        assert serving["performance"]["completion_tokens"] == 128
        assert serving["sampling"]["concurrency"] == 4
        assert serving["runtime"]["hybrid_config"] == "fixture-v1"
        assert state["peak"] >= 2 and state["gets"] == 2
        assert len(state["requests"]) == 10
        repeated = json.loads(command(base, submit, folder, "eval-idempotent"))
        assert repeated["id"] == run_id and len(state["requests"]) == 10
        attempt = Path(repeated["path"]) / "attempts" / "0001"
        assert not (attempt / "checkpoint").exists(), "serving eval must not copy full model weights"
        identity = json.loads((attempt / "checkpoint-identity.json").read_text())
        assert identity == serving["checkpoint_identity"]
        assert (attempt / "prompts.jsonl").read_bytes() == prompts.read_bytes()
        traces = [json.loads(line) for line in (attempt / "requests.jsonl").read_text().splitlines()]
        assert len(traces) == 8 and all(row["phase"] == "measure" for row in traces)
        assert all(row["sampling"]["temperature"] == 0 for row in traces)
        warmup = [json.loads(line) for line in (attempt / "warmup.jsonl").read_text().splitlines()]
        assert len(warmup) == 2 and all(row["phase"] == "warmup" for row in warmup)
        assert all(row["sampling"]["temperature"] == 0 for row in warmup)
        calibrate_config = folder / "calibrate.json"
        calibration = dict(traces=str(attempt / "requests.jsonl"), stride=4, max_iter=20,
            provenance=dict(target="fixture-model-v1", tokenizer="fixture-tokenizer-v1",
                            runtime="fixture-runtime-v1", dataset="fixture-data-v1"))
        write(calibrate_config, calibration)
        command(base, ["calibrate", "-c", str(calibrate_config), "--plan"], folder, "calibrate-plan")
        submit_calibrate = ["calibrate", "-c", str(calibrate_config), "--request-id", "calibration-fixture", "--json"]
        calibration_id = json.loads(command(base, submit_calibrate, folder, "calibrate-submit"))["id"]
        fitted = watch(base, calibration_id, folder, "calibrate")
        artifact_path = Path(fitted["calibrator"])
        artifact = json.loads(artifact_path.read_text())
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == fitted["calibrator_sha256"]
        assert artifact["provenance"]["temperature"] == 0
        assert len(artifact["coefficients"]["intercepts"]) == 15
        for split in ("training", "validation"):
            assert artifact["fit"][split]["labels"] > 0
            assert math.isfinite(artifact["fit"][split]["log_loss"])
        assert json.loads(command(base, submit_calibrate, folder, "calibrate-idempotent"))["id"] == calibration_id
        for failure, expected_error in (("missing-cycles", "spec_verify_cycles"),
                                        ("malformed-output", "output token IDs")):
            state["failure"] = failure
            submit_failed = ["eval", str(checkpoint), "-c", str(config), "--request-id", failure, "--json"]
            failed_id = json.loads(command(base, submit_failed, folder, failure + "-submit"))["id"]
            failed = watch(base, failed_id, folder, failure, succeeds=False)
            assert failed["complete"] is False and expected_error in failed["error"], failed
        bad_config = folder / "warmup-calibrate.json"
        write(bad_config, dict(calibration, traces=str(attempt / "warmup.jsonl")))
        warmup_id = json.loads(command(base, ["calibrate", "-c", str(bad_config), "--json"], folder,
                                      "warmup-calibrate-submit"))["id"]
        command(base, ["watch", warmup_id, "--until", "terminal", "--timeout", "60",
                      "--interval", "0.1", "--jsonl"], folder, "warmup-calibrate-watch", expected=1)
        status = json.loads(command(base, ["show", warmup_id, "--json"], folder, "warmup-calibrate-show"))
        assert status["state"] == "failed"
        sampled_traces = json.loads(json.dumps(traces))
        sampled_traces[0]["sampling"]["temperature"] = 0.7
        sampled_path = folder / "sampled-traces.jsonl"
        sampled_path.write_text("".join(json.dumps(row) + "\n" for row in sampled_traces))
        sampled_config = folder / "sampled-calibrate.json"
        write(sampled_config, dict(calibration, traces=str(sampled_path)))
        sampled_id = json.loads(command(base, ["calibrate", "-c", str(sampled_config), "--json"], folder,
                                       "sampled-calibrate-submit"))["id"]
        command(base, ["watch", sampled_id, "--until", "terminal", "--timeout", "60",
                      "--interval", "0.1", "--jsonl"], folder, "sampled-calibrate-watch", expected=1)
        sampled_status = json.loads(command(base, ["show", sampled_id, "--json"], folder,
                                            "sampled-calibrate-show"))
        assert sampled_status["state"] == "failed"
        sampled_attempt = Path(sampled_status["path"]) / "attempts" / "0001"
        assert "trace-level temperature=0" in (sampled_attempt / "worker.log").read_text()
        assert not (sampled_attempt / "calibrator.json").exists()
        report.update(passed=True, eval_run_id=run_id, calibration_run_id=calibration_id,
                      measured_requests=8, warmup_requests=2, peak_http_concurrency=state["peak"],
                      serving_copied_checkpoint=False, checkpoint_identity_persisted=True,
                      expected_failures=["missing-cycles", "malformed-output", "warmup-calibration",
                                         "sampled-calibration"],
                      calibrator_sha256=fitted["calibrator_sha256"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        write(folder / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
