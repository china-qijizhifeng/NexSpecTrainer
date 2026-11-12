"""Measure native V4.1 target/DSPARK/Hybrid at the 64K context boundary.

All three servers run sequentially in one allocation. Each measured request has
65535, 65536 or 65537 officially encoded input tokens. The existing small-suite
calibrator is explicitly reused, not represented as a fit on long-context data.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import requests

from full_model_acceptance import compare_outputs, nsp_run, run, sha, stop, write
from native_cycle_hook import DIRECTORY_ENV, install, load_cycles
from specforge.hybrid.config import HybridConfig
from specforge.nsp.eval_inputs import serving_checkpoint_identity


INPUT_LENGTHS = (65535, 65536, 65537)
CONTEXT_LENGTH = 69632
TOTAL_TOKENS = 98304
PREFILL_CHUNK = 4096
MAX_NEW_TOKENS = 512
CALIBRATOR_SHA256 = "0f481ca190c4312613fc191ffc94af292407b2f6c3e69f0a4a09bc8b2b8a7d09"


def prepare_prompts(model, output, *, encode=None):
    if encode is None:
        from transformers import AutoTokenizer
        from sglang.srt.entrypoints.openai.chat_encoding import encode_simple_chat

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
        def encode(content, thinking):
            return encode_simple_chat(tokenizer=tokenizer, spec="dsv41",
                messages=[dict(role="user", content=content)], thinking_mode=thinking)
    rows = []
    for size in INPUT_LENGTHS:
        code = "\n".join(f"values_ctx{size}_{j} = [{j}, {j+1}, {j+2}]" for j in range(24)) + "\n"
        repeated = "\n".join(["print('alpha beta gamma delta epsilon')"] * 24) + "\n"
        for suite in ("near-code-copy", "far-code-copy", "repeated-pattern", "reasoning-no-match"):
            instruction = ("Return exactly the marked Python source, without Markdown or explanation. "
                           "Preserve all lines verbatim.\n")
            block = "SOURCE START\n" + (repeated if suite == "repeated-pattern" else code) + "SOURCE END\n"
            if suite == "far-code-copy":
                start, end = instruction + block + "\nUnrelated reference context:\n", "\nReturn the marked source above.\n"
            elif suite == "reasoning-no-match":
                start = "Unrelated reference context follows.\n"
                end = "\nIgnore that reference context. What is 37 multiplied by 19? Give a short calculation.\n"
            else:
                start, end = instruction + "\nUnrelated reference context:\n", "\n" + block
            thinking = "thinking" if suite == "reasoning-no-match" else "chat"
            padding = size
            for _ in range(12):
                ids = encode(start + " reference" * padding + end, thinking)
                difference = size - len(ids)
                if not difference:
                    break
                padding += difference
                if padding < 0:
                    raise ValueError("64K prompt cannot fit its fixed source")
            if len(ids) != size or any(type(token) is not int or token < 0 for token in ids):
                raise ValueError("official encoding did not reach the exact requested token count")
            rows.append(dict(id=f"{suite}-{size}", suite=suite, thinking_mode=thinking,
                             input_ids=ids, source_placement="before-padding" if suite == "far-code-copy" else "after-padding"))
    path = output / "measured-prompts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write(output / "inputs.json", dict(count=len(rows), input_lengths=list(INPUT_LENGTHS),
        encoder="sglang.encode_simple_chat:dsv41", prompts_sha256=sha(path),
        samples=[{k: v for k, v in row.items() if k != "input_ids"} | {"input_tokens": len(row["input_ids"])} for row in rows],
        scope="synthetic 64K engineering suite; repeated reference filler; not a representative workload benchmark"))
    return rows


def freeze_calibrator(model, artifact, folder):
    if sha(artifact) != CALIBRATOR_SHA256:
        raise ValueError("64K acceptance requires the frozen r4 calibrator")
    raw = json.loads(artifact.read_text())
    provenance = raw["provenance"]
    if (provenance.get("target") != sha(model / "config.json")
            or provenance.get("tokenizer") != sha(model / "tokenizer.json")
            or provenance.get("temperature") != 0):
        raise ValueError("calibrator target/tokenizer/sampling identity differs")
    destination = folder / "reused-ngram-calibrator.json"
    shutil.copy2(artifact, destination)
    HybridConfig(calibrator=str(destination)).coefficients()
    write(folder / "calibrator-reuse.json", dict(source=str(artifact), sha256=sha(destination),
        original_provenance=provenance,
        scope="small-suite r4 calibrator reused unchanged; no 64K calibration-quality claim"))
    return destination


def server_command(model, variant, port):
    command = ["sglang", "serve", "--model-path", str(model), "--trust-remote-code",
        "--tp-size", "8", "--ep-size", "8", "--attention-backend", "dsv4",
        "--moe-runner-backend", "flashinfer_mxfp4", "--engram-config", '{"placement":"cpu"}',
        "--mem-fraction-static", "0.8", "--context-length", str(CONTEXT_LENGTH),
        "--max-total-tokens", str(TOTAL_TOKENS), "--swa-full-tokens-ratio", "0.8",
        "--max-prefill-tokens", str(PREFILL_CHUNK), "--max-running-requests", "1",
        "--disable-radix-cache", "--skip-tokenizer-init", "--chunked-prefill-size", str(PREFILL_CHUNK),
        "--no-enable-decoder-swa-bounded-replay", "--cuda-graph-backend-prefill", "disabled",
        "--cuda-graph-backend-decode", "full", "--cuda-graph-bs-decode", "1",
        "--decode-log-interval", "1", "--random-seed", "42", "--host", "127.0.0.1", "--port", str(port)]
    if variant != "target":
        command += ["--speculative-draft-model-path", str(model), "--speculative-algorithm", "DSPARK",
                    "--speculative-dspark-block-size", "5"]
    return command


def launch(model, variant, folder, calibrator, port, timeout):
    try:
        requests.get(f"http://127.0.0.1:{port}/get_server_info", timeout=2)
    except requests.RequestException:
        pass
    else:
        raise RuntimeError("dedicated acceptance port is occupied")
    env = os.environ.copy()
    for key in (DIRECTORY_ENV, "SGLANG_DSPARK_HYBRID_CONFIG", "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK"):
        env.pop(key, None)
    env.update(SGLANG_RAGGED_VERIFY_MODE="static", SGLANG_DSPARK_USE_GREEDY_DRAFT="1",
               SGLANG_JIT_DEEPGEMM_FAST_WARMUP="1")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", "")
    if variant != "target":
        (folder / "verify-cycles").mkdir()
        env[DIRECTORY_ENV] = str(folder / "verify-cycles")
    if variant == "hybrid":
        path = folder / "hybrid-config.json"
        write(path, dict(mode="hybrid", calibrator=str(calibrator), extra_budget_per_request=15, slot_price=0.0))
        env["SGLANG_DSPARK_HYBRID_CONFIG"] = str(path)
    if any("SIMULATE" in key and value not in ("", "0", "false", "False") for key, value in env.items()):
        raise ValueError("simulation settings must be absent")
    command = server_command(model, variant, port)
    write(folder / "launch.json", dict(command=command, launch_timeout_seconds=timeout,
        environment={key: value for key, value in env.items() if key.startswith(("SGLANG_", "PYTHONPATH", "HYBRID_ACCEPTANCE_"))}))
    log = (folder / "server.log").open("x")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    write(folder / "server-process.json", dict(pid=process.pid))
    started = time.monotonic()
    try:
        while time.monotonic() - started < timeout:
            if process.poll() is not None:
                raise RuntimeError(f"{variant} server exited {process.returncode}; inspect server.log")
            try:
                response = requests.get(f"http://127.0.0.1:{port}/get_server_info", timeout=5)
                if response.ok:
                    write(folder / "server-info.json", response.json())
                    write(folder / "readiness.json", dict(seconds=time.monotonic() - started))
                    return process, log
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError(f"{variant} readiness exceeded {timeout} seconds")
    except BaseException:
        stop(process, log)
        raise


def evaluate(base, model, variant, folder, inputs, port):
    expected = dict(model_path=str(model), random_seed=42, context_length=CONTEXT_LENGTH,
        chunked_prefill_size=PREFILL_CHUNK, max_running_requests=1,
        speculative_algorithm=None if variant == "target" else "DSPARK",
        speculative_draft_model_path=None if variant == "target" else str(model))
    if variant != "target":
        expected["speculative_num_draft_tokens"] = 16 if variant == "hybrid" else 6
    spec = dict(metrics=["performance"] if variant == "target" else ["al", "performance"], seed=42,
        serving=dict(protocol="native", url=f"http://127.0.0.1:{port}", model="deepseek-v4.1-flash",
            checkpoint_field="model_path", prompts=str(inputs), max_new_tokens=MAX_NEW_TOKENS,
            timeout=2400, concurrency=1, warmup_requests=1, expected_server_args=expected))
    config = folder / "measured-eval.json"
    write(config, spec)
    return nsp_run(base, ["eval", str(model), "-c", str(config), "--name", variant + "-64k",
        "--request-id", variant + "-64k"], folder, "measured")


def summarize_cycles(observed, prompts):
    suites = {row["id"]: row["suite"] for row in prompts}
    groups = {}
    continued = []
    for row in observed:
        entry = groups.setdefault(suites[row["sample_id"]], dict(requests=0, cycles=0, output_tokens=0,
            verify_committed_tokens=0, maximum_commit=0, ctx_min=None, ctx_max=None))
        entry["requests"] += 1
        entry["cycles"] += len(row["cycles"])
        emitted = sum(c["committed_output_tokens"] for c in row["cycles"])
        entry["verify_committed_tokens"] += emitted
        entry["output_tokens"] += row["prefill_tokens"] + emitted
        for index, cycle in enumerate(row["cycles"]):
            ctx = cycle["ctx_len"]
            if ctx < min(INPUT_LENGTHS) + 1:
                raise ValueError("64K observation unexpectedly has a short context")
            entry["ctx_min"] = ctx if entry["ctx_min"] is None else min(entry["ctx_min"], ctx)
            entry["ctx_max"] = ctx if entry["ctx_max"] is None else max(entry["ctx_max"], ctx)
            entry["maximum_commit"] = max(entry["maximum_commit"], cycle["committed_output_tokens"])
            if cycle["committed_output_tokens"] > 6 and index + 1 < len(row["cycles"]):
                next_cycle = row["cycles"][index + 1]
                if next_cycle["ctx_len"] != ctx + cycle["committed_output_tokens"]:
                    raise ValueError("64K long commit did not advance the next context correctly")
                if next_cycle["committed_output_tokens"] > 0:
                    continued.append(dict(sample_id=row["sample_id"], commit=cycle, next_cycle=next_cycle))
    for entry in groups.values():
        entry["al"] = entry["output_tokens"] / entry["cycles"] if entry["cycles"] else None
        entry["mean_verify_commit"] = entry["verify_committed_tokens"] / entry["cycles"] if entry["cycles"] else None
    return dict(suites=groups, total_cycles=sum(x["cycles"] for x in groups.values()),
        long_commits_followed_by_decode=continued,
        scope="actual context per verify cycle; AL includes initial prefill token, mean_verify_commit excludes it")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("RUN_DIR", ".")) / "full-model")
    parser.add_argument("--model", type=Path, default=Path("/gpfs/shared-model-weight/models/deepseek-ai/DeepSeek-V4.1-Flash-1017"))
    parser.add_argument("--runtime-bundle", required=True, type=Path)
    parser.add_argument("--calibrator", required=True, type=Path)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--launch-timeout-seconds", type=int, default=5400)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if max(INPUT_LENGTHS) + MAX_NEW_TOKENS > CONTEXT_LENGTH or TOTAL_TOKENS < CONTEXT_LENGTH:
        raise ValueError("64K profile cannot fit the declared prompt/output bounds")
    if args.launch_timeout_seconds <= 0:
        parser.error("launch timeout must be positive")
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    prompts = prepare_prompts(args.model, folder)
    calibrator = freeze_calibrator(args.model, args.calibrator, folder)
    identity = serving_checkpoint_identity(args.model)
    write(folder / "checkpoint-identity.json", identity)
    write(folder / "profile.json", dict(context_length=CONTEXT_LENGTH, max_total_tokens=TOTAL_TOKENS,
        chunked_prefill_size=PREFILL_CHUNK, max_prefill_tokens=PREFILL_CHUNK, concurrency=1,
        max_new_tokens=MAX_NEW_TOKENS, input_lengths=list(INPUT_LENGTHS), variants=["target", "dspark", "hybrid"],
        manifest_sha256=sha(args.runtime_bundle / "manifest.json"), calibrator_sha256=sha(calibrator),
        commands={v: server_command(args.model, v, args.port) for v in ("target", "dspark", "hybrid")}))
    if args.preflight_only:
        print(json.dumps(dict(preflight_passed=True, samples=len(prompts), output_dir=str(folder))))
        return
    sglang_source = Path(importlib.util.find_spec("sglang").origin).parent
    patch = [sys.executable, "-m", "specforge.hybrid.runtime_patch", "--root", str(sglang_source.parent.parent),
             "--bundle", str(args.runtime_bundle.resolve())]
    run(patch, folder, "sglang-patch-plan")
    run(patch + ["--execute"], folder, "sglang-patch-apply")
    install(sglang_source, folder / "native-cycle-instrumentation.json")
    base = [sys.executable, "-m", "specforge.nsp.cli", "--root", str(folder / "nsp")]
    outputs, results, cycles, comparisons = {}, {}, {}, {}
    passed = False
    error = None
    try:
        for variant in ("target", "dspark", "hybrid"):
            destination = folder / variant
            destination.mkdir()
            process, log = launch(args.model, variant, destination, calibrator, args.port, args.launch_timeout_seconds)
            try:
                _, result, attempt = evaluate(base, args.model, variant, destination,
                    folder / "measured-prompts.jsonl", args.port)
                results[variant] = result["stages"]["serving"]
                outputs[variant] = list(map(json.loads, (attempt / "requests.jsonl").read_text().splitlines()))
                if variant != "target":
                    observed = [load_cycles(destination / "verify-cycles", row, 15 if variant == "hybrid" else 5)
                                for row in outputs[variant]]
                    write(destination / "observed-measured-cycles.json", observed)
                    cycles[variant] = summarize_cycles(observed, prompts)
                    write(destination / "cycle-summary.json", cycles[variant])
                if serving_checkpoint_identity(args.model) != identity:
                    raise ValueError("checkpoint inventory changed during 64K evaluation")
                if variant != "target":
                    comparisons["target-vs-" + variant] = compare_outputs(outputs["target"], outputs[variant], folder / ("target-vs-" + variant + ".json"))
            finally:
                stop(process, log)
        comparisons["dspark-vs-hybrid"] = compare_outputs(outputs["dspark"], outputs["hybrid"], folder / "dspark-vs-hybrid.json")
        if not cycles["hybrid"]["long_commits_followed_by_decode"]:
            raise RuntimeError("64K Hybrid has no long commit followed by continued decoding")
        if not all(comparisons.values()):
            raise RuntimeError("64K strict T=0 output equality failed; original receipts retained")
        passed = True
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        raise
    finally:
        write(folder / "summary.json", dict(passed=passed, error=error, results=results, comparisons=comparisons,
            cycles=cycles, all_variants_same_allocation=True, temperature=0,
            claim_limit="12 synthetic 64K prompts; unchanged short-suite calibrator; no representative workload or production claim"))
    print(json.dumps(dict(passed=passed, output_dir=str(folder))))


if __name__ == "__main__":
    main()
