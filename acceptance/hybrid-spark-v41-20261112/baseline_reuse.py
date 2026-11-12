"""Reuse only the frozen r4 correctness receipts, without a speed comparison.

The allowed runtime change retains the trained confidence head for Hybrid in
STATIC mode. The nine existing patch targets and the evaluation code must be
identical. Calibration provenance continues to name the original runtime.
"""

import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from specforge.nsp.eval_inputs import serving_checkpoint_identity


BASELINE_COMMIT = "d834b94fef28086490c7c62e3ee5cfc250fcf45d"
BASELINE_MANIFEST = "bffb69ef21d01b27cfe4c906c3b75cc4e7593e5c2722e23ee0f3c68932ea7f10"
MODEL_FILE = "python/sglang/srt/models/deepseek_v4_dspark.py"
BUNDLE = Path("patches/sglang/v41-hybrid-spark-38d9e4a3fe")
ACCEPTANCE = Path("acceptance/hybrid-spark-v41-20261112")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError("baseline reuse rejected: " + message)


def load_run(folder, variant, label, identity, prompts):
    status = read(folder / variant / (label + "-show.stdout"))
    require(status["state"] == "succeeded", variant + " " + label + " did not succeed")
    attempt = folder / "nsp/runs" / status["id"] / "attempts/0001"
    result = read(attempt / "results.json")
    require(result == status["completion"] and result["complete"], "inconsistent nsp completion")
    serving = result["stages"]["serving"]
    require(serving["complete"] and serving["checkpoint_identity"] == identity, "checkpoint identity changed")
    require(serving["prompts_sha256"] == sha(prompts), "prompt receipt differs")
    expected = {row["id"]: row["input_ids"] for row in map(json.loads, prompts.read_text().splitlines())}
    rows = list(map(json.loads, (attempt / "requests.jsonl").read_text().splitlines()))
    require(len(rows) == len(expected) == serving["request_count"], "request coverage differs")
    require({row["sample_id"]: row["input_ids"] for row in rows} == expected, "request input IDs differ")
    require(all(row["sampling"]["temperature"] == 0 and row["output_ids"] for row in rows), "invalid greedy output receipt")
    require(serving["sampling"] == dict(temperature=0.0, top_p=1.0, max_new_tokens=256,
        seed=42, concurrency=4, protocol="native", warmup_requests=2), "sampling recipe changed")
    launch = read(folder / variant / "launch.json")
    require(launch["environment"]["SGLANG_RAGGED_VERIFY_MODE"] == "static", "baseline must use STATIC")
    require(not launch["environment"].get("SGLANG_DSPARK_HYBRID_CONFIG"), "baseline unexpectedly enabled Hybrid")
    require(serving["runtime"]["random_seed"] == 42, "baseline server seed changed")
    require(status["manifest"]["commit"] == BASELINE_COMMIT, "baseline evaluation commit differs")
    return status, serving, rows, attempt


def reuse_baseline(folder, source, output, current_source, bundle, model):
    folder, source, output, current_source, bundle, model = map(Path,
        (folder, source, output, current_source, bundle, model))
    # Git is used only to attest the immutable source. No checkout is modified.
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    require(revision == BASELINE_COMMIT, "source is not the supported r4 commit")
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    require(not dirty, "baseline source has tracked modifications")
    old_manifest = source / BUNDLE / "manifest.json"
    require(sha(old_manifest) == BASELINE_MANIFEST, "old runtime bundle fingerprint differs")
    previous, current = read(old_manifest), read(bundle / "manifest.json")
    require(previous["base_revision"] == current["base_revision"], "SGLang base differs")
    require(len(previous["files"]) == 9 and set(current["files"]) == set(previous["files"]) | {MODEL_FILE}, "unsupported patch scope")
    require(all(current["files"][name] == pins for name, pins in previous["files"].items()), "existing runtime patch target changed")
    require(current["files"][MODEL_FILE] == dict(
        before_sha256="55edc39a44f4e8af5621893a632988947378a80e841771a3c3b21cd49df26b75",
        after_sha256="10586ccf1f393233a8a3ffe4e3d59614ceb7a58f872afe47c91af82f9165809f"),
        "confidence builder differs from the verified opt-in change")
    helpers = [ACCEPTANCE / "native_cycle_hook.py"]
    for package in ("specforge/nsp", "specforge/hybrid"):
        paths = {p.relative_to(source) for p in (source / package).rglob("*.py")}
        require(paths == {p.relative_to(current_source) for p in (current_source / package).rglob("*.py")}, "helper file inventory changed")
        helpers.extend(sorted(paths))
    require(all(sha(source / path) == sha(current_source / path) for path in helpers), "nsp, runtime or observer code changed")
    function_names = ("write", "sha", "run", "nsp_run", "prepare_prompts", "launch", "stop",
                      "evaluate", "compare_outputs", "diagnose_reasoning")
    def functions(path):
        return {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
    old_functions = functions(source / ACCEPTANCE / "full_model_acceptance.py")
    new_functions = functions(current_source / ACCEPTANCE / "full_model_acceptance.py")
    require(all(old_functions[name] == new_functions[name] for name in function_names), "measurement or prompt-generation code changed")
    require(read(folder / "environment.json")["bundle_manifest_sha256"] == BASELINE_MANIFEST, "baseline runtime receipt differs")
    for name in ("calibration-prompts.jsonl", "measured-prompts.jsonl", "inputs.json"):
        require(sha(folder / name) == sha(output / name), "regenerated frozen inputs differ: " + name)
    identity = serving_checkpoint_identity(model)
    outputs, results, run_ids = {}, {}, {}
    snapshots = output / "baseline-receipts"
    snapshots.mkdir()
    for variant in ("target", "dspark"):
        status, serving, rows, attempt = load_run(folder, variant, "measured", identity, folder / "measured-prompts.jsonl")
        outputs[variant], results[variant], run_ids[variant] = rows, serving, status["id"]
        dest = snapshots / variant
        dest.mkdir()
        for name in ("results.json", "requests.jsonl", "checkpoint-identity.json", "manifest.json", "completion.json"):
            shutil.copy2(attempt / name, dest / name)
        shutil.copytree(folder / variant / "diagnostics", dest / "diagnostics")
        for name in ("launch.json", "server-info.json", "measured-show.stdout"):
            shutil.copy2(folder / variant / name, dest / name)
        if variant == "dspark":
            shutil.copy2(folder / variant / "observed-measured-cycles.json", dest / "observed-measured-cycles.json")
    _, _, _, traces = load_run(folder, "target", "calibration-traces", identity, folder / "calibration-prompts.jsonl")
    fit = read(folder / "target/calibrate-results.stdout")
    status = read(folder / "target/calibrate-show.stdout")
    require(status["state"] == "succeeded" and fit["complete"] and status["completion"] == fit, "calibration Run incomplete")
    artifact = folder / "nsp/runs" / status["id"] / "attempts/0001/ngram-calibrator.json"
    require(sha(artifact) == fit["calibrator_sha256"], "calibrator content changed")
    provenance = read(artifact)["provenance"]
    require(provenance == dict(target=sha(model / "config.json"), tokenizer=sha(model / "tokenizer.json"),
        runtime=BASELINE_MANIFEST, dataset=sha(folder / "calibration-prompts.jsonl"),
        traces_sha256=sha(traces / "requests.jsonl"), temperature=0), "calibrator provenance differs")
    calibration = output / "reused-ngram-calibrator.json"
    shutil.copy2(artifact, calibration)
    receipt = dict(baseline_dir=str(folder.resolve()), baseline_source=str(source.resolve()),
        baseline_commit=revision, baseline_runtime_manifest_sha256=BASELINE_MANIFEST,
        current_runtime_manifest_sha256=sha(bundle / "manifest.json"),
        evaluated_runs=run_ids, calibration_run=status["id"], calibrator_sha256=sha(calibration),
        calibration_provenance=provenance, checkpoint_identity=identity,
        identical_helpers={str(path): sha(source / path) for path in helpers},
        scope="correctness receipts from the preceding allocation; no cross-allocation throughput comparison")
    (output / "baseline-reuse.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return outputs, results, calibration
