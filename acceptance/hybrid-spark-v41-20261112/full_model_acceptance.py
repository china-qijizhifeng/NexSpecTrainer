"""Run native V4.1 target/DSPARK/Hybrid differential gates inside one GPU Run.

All evaluation and calibration work is submitted through nsp. Model launches
belong to this dedicated acceptance allocation and are stopped in finally.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

from native_cycle_hook import DIRECTORY_ENV, install, load_cycles


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(argv, folder, label, *, env=None):
    result = subprocess.run(argv, env=env, capture_output=True, text=True)
    (folder / (label + ".stdout")).write_text(result.stdout)
    (folder / (label + ".stderr")).write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{label} failed with exit {result.returncode}; see its persistent logs")
    return result.stdout


def nsp_run(base, argv, folder, label):
    run(base + argv + ["--plan"], folder, label + "-plan")
    status = json.loads(run(base + argv + ["--json"], folder, label + "-submit"))
    run(base + ["watch", status["id"], "--until", "terminal", "--interval", "1",
                "--timeout", "10800", "--jsonl"], folder, label + "-watch")
    status = json.loads(run(base + ["show", status["id"], "--json"], folder, label + "-show"))
    if status["state"] != "succeeded":
        raise RuntimeError(f"{label} ended in {status['state']}")
    result = json.loads(run(base + ["results", status["id"], "--json"], folder, label + "-results"))
    if not result["complete"]:
        raise RuntimeError(f"{label} did not publish complete results")
    return status, result, Path(status["path"]) / "attempts" / "0001"


def prepare_prompts(model, output):
    from transformers import AutoTokenizer
    from sglang.srt.entrypoints.openai.chat_encoding import encode_simple_chat

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    calibration, measured = [], []
    for group, count in ((calibration, 24), (measured, 12)):
        for index in range(count):
            prefix = "calibration" if group is calibration else "heldout"
            code = "\n".join(f"values_{prefix}_{index}_{j} = [{j}, {j+1}, {j+2}]" for j in range(18))
            content = ("Return exactly the following Python source, with no explanation or Markdown. "
                       "Preserve the text and all lines verbatim.\n\n" + code + "\n")
            ids = encode_simple_chat(tokenizer=tokenizer, spec="dsv41",
                messages=[dict(role="user", content=content)], thinking_mode="chat")
            if not isinstance(ids, list) or not ids or len(ids) > 4096:
                raise ValueError("unexpected native dsv41 recipe output")
            group.append(dict(id=f"{prefix}-{index}", suite="exact-code-copy", input_ids=ids))
    edge_prompts = [
        ("short-copy-eos", "short-copy-eos", "Return only this Python statement, without Markdown or explanation:\nanswer = 42\n"),
        ("repeated-pattern", "repeated-pattern", "Return this Python source verbatim, without Markdown or explanation:\n" +
            "\n".join(["print('alpha beta gamma delta epsilon')"] * 8) + "\n"),
    ]
    for sample_id, suite, content in edge_prompts:
        measured.append(dict(id=sample_id, suite=suite, input_ids=encode_simple_chat(
            tokenizer=tokenizer, spec="dsv41", messages=[dict(role="user", content=content)], thinking_mode="chat")))
    for target_length in (4095, 4096):
        start = "Return only the final source snippet verbatim, without Markdown or explanation.\nReference context:\n"
        end = "\nFinal source snippet:\ncopy_marker = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']\n"
        padding = target_length
        for _ in range(8):
            ids = encode_simple_chat(tokenizer=tokenizer, spec="dsv41", thinking_mode="chat",
                messages=[dict(role="user", content=start + " context" * padding + end)])
            difference = target_length - len(ids)
            if not difference:
                break
            padding += difference
            if padding < 0:
                raise ValueError("cannot construct boundary prompt")
        if len(ids) != target_length:
            raise ValueError("native recipe boundary prompt length mismatch")
        measured.append(dict(id=f"long-copy-{target_length}", suite="long-copy-boundary", input_ids=ids))
    measured += [dict(id="reasoning-no-match", suite="reasoning-thinking",
        input_ids=encode_simple_chat(tokenizer=tokenizer, spec="dsv41",
            messages=[dict(role="user", content="What is 37 multiplied by 19? Give a short calculation.")],
            thinking_mode="thinking"))]
    for name, rows in (("calibration-prompts", calibration), ("measured-prompts", measured)):
        path = output / (name + ".jsonl")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write(output / "inputs.json", dict(calibration_count=len(calibration), measured_count=len(measured),
        encoder="sglang.encode_simple_chat:dsv41", copy_thinking_mode="chat", reasoning_thinking_mode="thinking",
        measured_prompt_lengths={row["id"]: len(row["input_ids"]) for row in measured},
        calibration_sha256=sha(output / "calibration-prompts.jsonl"),
        measured_sha256=sha(output / "measured-prompts.jsonl"),
        scope="small engineering acceptance dataset; no model-quality or heldout calibration coverage claim"))


def launch(model, variant, folder, hybrid_config, port, launch_timeout_seconds):
    try:
        response = requests.get(f"http://127.0.0.1:{port}/get_server_info", timeout=2)
    except requests.RequestException:
        response = None
    if response is not None:
        raise RuntimeError("dedicated acceptance port is already occupied")
    command = ["sglang", "serve", "--model-path", str(model), "--trust-remote-code",
        "--tp-size", "8", "--ep-size", "8", "--attention-backend", "dsv4",
        "--moe-runner-backend", "flashinfer_mxfp4", "--engram-config", '{"placement":"cpu"}',
        "--mem-fraction-static", "0.8", "--context-length", "8192", "--max-total-tokens", "32768",
        "--swa-full-tokens-ratio", "0.8", "--max-prefill-tokens", "8192",
        "--max-running-requests", "4", "--disable-radix-cache", "--skip-tokenizer-init",
        "--chunked-prefill-size", "-1", "--no-enable-decoder-swa-bounded-replay",
        "--cuda-graph-backend-prefill", "disabled", "--cuda-graph-backend-decode", "full",
        "--cuda-graph-bs-decode", "1", "2", "3", "4", "--decode-log-interval", "1",
        "--host", "127.0.0.1", "--port", str(port)]
    if variant != "target":
        command += ["--speculative-draft-model-path", str(model), "--speculative-algorithm", "DSPARK",
                    "--speculative-dspark-block-size", "5"]
    env = os.environ.copy()
    for key in ("SGLANG_DSPARK_HYBRID_CONFIG", "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK"):
        env.pop(key, None)
    env["SGLANG_RAGGED_VERIFY_MODE"] = "static"
    env["SGLANG_DSPARK_USE_GREEDY_DRAFT"] = "1"
    env["SGLANG_JIT_DEEPGEMM_FAST_WARMUP"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", "")
    if variant != "target":
        cycles = folder / "verify-cycles"
        cycles.mkdir()
        env[DIRECTORY_ENV] = str(cycles)
    if hybrid_config:
        env["SGLANG_DSPARK_HYBRID_CONFIG"] = str(hybrid_config)
    if any("SIMULATE" in key and value not in ("", "0", "false", "False") for key, value in env.items()):
        raise ValueError("simulation environment must be absent")
    write(folder / "launch.json", dict(command=command, launch_timeout_seconds=launch_timeout_seconds,
        environment={key: value for key, value in env.items()
        if key.startswith(("SGLANG_", "PYTHONPATH", "PYTORCH_"))}))
    log = (folder / "server.log").open("x")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    write(folder / "server-process.json", dict(pid=process.pid))
    started = time.monotonic()
    while time.monotonic() - started < launch_timeout_seconds:
        if process.poll() is not None:
            stop(process, log)
            raise RuntimeError(f"{variant} server exited {process.returncode}; inspect {folder / 'server.log'}")
        try:
            info = requests.get(f"http://127.0.0.1:{port}/get_server_info", timeout=5)
            if info.ok:
                write(folder / "server-info.json", info.json())
                write(folder / "readiness.json", dict(seconds=time.monotonic() - started))
                return process, log
        except requests.RequestException:
            pass
        time.sleep(2)
    stop(process, log)
    raise TimeoutError(f"{variant} readiness exceeded {launch_timeout_seconds} seconds")


def stop(process, log):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
    log.close()


def evaluate(base, model, variant, folder, inputs, port, label):
    args = dict(model_path=str(model), speculative_algorithm=None if variant == "target" else "DSPARK",
                speculative_draft_model_path=None if variant == "target" else str(model))
    spec = dict(metrics=["performance"] if variant == "target" else ["al", "performance"], seed=42,
        serving=dict(protocol="native", url=f"http://127.0.0.1:{port}", model="deepseek-v4.1-flash",
            checkpoint_field="model_path", prompts=str(inputs), max_new_tokens=256, timeout=2400,
            concurrency=4, warmup_requests=2, expected_server_args=args))
    config = folder / (label + "-eval.json")
    write(config, spec)
    return nsp_run(base, ["eval", str(model), "-c", str(config), "--name", variant + "-" + label,
        "--request-id", variant + "-" + label], folder, label)


def compare_outputs(baseline, candidate, output):
    a = {row["sample_id"]: row["output_ids"] for row in baseline}
    b = {row["sample_id"]: row["output_ids"] for row in candidate}
    differences = []
    if a.keys() != b.keys():
        raise ValueError("variant request IDs differ")
    for sample_id, ids in a.items():
        other = b[sample_id]
        if ids != other:
            first = next((i for i, pair in enumerate(zip(ids, other)) if pair[0] != pair[1]), min(len(ids), len(other)))
            differences.append(dict(sample_id=sample_id, first_divergence=first,
                                    baseline_tokens=ids[first:first + 8], candidate_tokens=other[first:first + 8],
                                    baseline_length=len(ids), candidate_length=len(other)))
    write(output, dict(equal=not differences, requests=len(a), differences=differences))
    return not differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("RUN_DIR", ".")) / "full-model")
    parser.add_argument("--model", type=Path, default=Path("/gpfs/shared-model-weight/models/deepseek-ai/DeepSeek-V4.1-Flash-1017"))
    parser.add_argument("--runtime-bundle", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--launch-timeout-seconds", type=int, default=5400)
    args = parser.parse_args()
    if args.launch_timeout_seconds <= 0:
        parser.error("--launch-timeout-seconds must be positive")
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    sglang_source = Path(importlib.util.find_spec("sglang").origin).parent
    sglang_root = sglang_source.parent.parent
    patch = [sys.executable, "-m", "specforge.hybrid.runtime_patch", "--root", str(sglang_root),
             "--bundle", str(args.runtime_bundle.resolve())]
    run(patch, folder, "sglang-patch-plan")
    run(patch + ["--execute"], folder, "sglang-patch-apply")
    install(sglang_source, folder / "native-cycle-instrumentation.json")
    write(folder / "environment.json", dict(python=sys.version, executable=sys.executable,
        sglang_source=str(sglang_source), bundle_manifest_sha256=sha(args.runtime_bundle / "manifest.json"),
        model_config_sha256=sha(args.model / "config.json"), model_index_sha256=sha(args.model / "model.safetensors.index.json")))
    prepare_prompts(args.model, folder)
    base = [sys.executable, "-m", "specforge.nsp.cli", "--root", str(folder / "nsp")]
    outputs, results, observed = {}, {}, {}
    calibration_artifact = None
    passed = False
    try:
        for variant in ("target", "dspark", "hybrid"):
            variant_folder = folder / variant
            variant_folder.mkdir()
            hybrid_config = None
            if variant == "hybrid":
                hybrid_config = variant_folder / "hybrid-config.json"
                write(hybrid_config, dict(calibrator=str(calibration_artifact), mode="hybrid",
                    extra_budget_per_request=15, slot_price=0.0))
            process, log = launch(args.model, variant, variant_folder, hybrid_config, args.port,
                                  args.launch_timeout_seconds)
            try:
                if variant == "target":
                    _, _, attempt = evaluate(base, args.model, variant, variant_folder,
                                              folder / "calibration-prompts.jsonl", args.port, "calibration-traces")
                    config = variant_folder / "calibration.json"
                    write(config, dict(traces=str(attempt / "requests.jsonl"), stride=2, seed=42,
                        provenance=dict(target=sha(args.model / "config.json"), tokenizer=sha(args.model / "tokenizer.json"),
                                        runtime=sha(args.runtime_bundle / "manifest.json"), dataset=sha(folder / "calibration-prompts.jsonl"))))
                    _, fit, _ = nsp_run(base, ["calibrate", "-c", str(config), "--name", "v41-acceptance-calibration",
                                              "--request-id", "v41-acceptance-calibration"], variant_folder, "calibrate")
                    calibration_artifact = Path(fit["calibrator"])
                _, result, attempt = evaluate(base, args.model, variant, variant_folder,
                                              folder / "measured-prompts.jsonl", args.port, "measured")
                results[variant] = result["stages"]["serving"]
                outputs[variant] = [json.loads(line) for line in (attempt / "requests.jsonl").read_text().splitlines()]
                if variant != "target":
                    observed[variant] = [load_cycles(variant_folder / "verify-cycles", row,
                                                    15 if variant == "hybrid" else 5)
                                         for row in outputs[variant]]
                    write(variant_folder / "observed-measured-cycles.json", observed[variant])
            finally:
                stop(process, log)
            if variant == "dspark":
                compare_outputs(outputs["target"], outputs[variant], folder / "target-vs-dspark.json")
        target_equal = compare_outputs(outputs["target"], outputs["hybrid"], folder / "target-vs-hybrid.json")
        dspark_equal = compare_outputs(outputs["dspark"], outputs["hybrid"], folder / "dspark-vs-hybrid.json")
        cycles = [dict(request_id=row["request_id"], sample_id=row["sample_id"], **cycle)
                  for row in observed["hybrid"] for cycle in row["cycles"]]
        long_commits = [cycle for cycle in cycles if cycle["committed_output_tokens"] > 6]
        continued_long_commits = [dict(request_id=row["request_id"], sample_id=row["sample_id"],
                                      commit=cycle, next_cycle=row["cycles"][index + 1])
            for row in observed["hybrid"] for index, cycle in enumerate(row["cycles"][:-1])
            if cycle["committed_output_tokens"] > 6
               and row["cycles"][index + 1]["committed_output_tokens"] > 0]
        write(folder / "long-commits.json", dict(telemetry="independent native CPU observer receipts",
            total_cycles=len(cycles), beyond_mtp_commits=long_commits,
            long_commits_followed_by_decode=continued_long_commits))
        if not continued_long_commits:
            raise RuntimeError("missing Hybrid commit beyond MTP horizon followed by another decode cycle")
        if not target_equal or not dspark_equal:
            raise RuntimeError("T=0 output divergence recorded; compare all three receipts before correctness claims")
        passed = True
    finally:
        write(folder / "summary.json", dict(passed=passed, temperature=0, results=results,
            calibration_artifact=str(calibration_artifact),
            budget_scope="fixed 15 extra slots/request engineering gate; no AV throughput or calibration-quality claim"))
    print(json.dumps(dict(passed=passed, output_dir=str(folder))))


if __name__ == "__main__":
    main()
