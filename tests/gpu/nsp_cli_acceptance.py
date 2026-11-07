"""Run actual nsp train/watch/checkpoint/eval subprocesses on tiny CUDA weights."""

import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
import argparse

from tests.gpu.nsp_cli_fixture import create_fixture

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output", type=Path, required=True, help="New or empty absolute output directory"
)
ROOT = parser.parse_args().output
if not ROOT.is_absolute():
    parser.error("--output must be absolute")
if ROOT.exists() and any(ROOT.iterdir()):
    parser.error(
        "--output must be new or empty; acceptance never overwrites a prior run"
    )
ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(REPOSITORY)
os.environ["PYTHONPATH"] = (
    str(REPOSITORY) + os.pathsep + os.environ.get("PYTHONPATH", "")
)
os.environ["PYTHONUNBUFFERED"] = "1"
assert torch.cuda.is_available(), "This acceptance requires an allocated CUDA GPU"

STORE = ROOT / "runs"
BASE = [sys.executable, "-m", "specforge.nsp.cli", "--root", str(STORE)]
checks = []
owned = []


def call(name, args, expected=0, timeout=300, decode=True):
    start = time.monotonic()
    p = subprocess.run(BASE + args, capture_output=True, text=True, timeout=timeout)
    (ROOT / f"{name}.stdout").write_text(p.stdout)
    (ROOT / f"{name}.stderr").write_text(p.stderr)
    checks.append(
        {"name": name, "exit_code": p.returncode, "seconds": time.monotonic() - start}
    )
    print(json.dumps(checks[-1]), flush=True)
    assert p.returncode == expected, (
        name,
        p.returncode,
        p.stdout[-2500:],
        p.stderr[-2500:],
    )
    return json.loads(p.stdout) if decode else p.stdout


def wait(name, run, expected=0):
    call(
        name,
        [
            "watch",
            run,
            "--until",
            "terminal",
            "--jsonl",
            "--interval",
            "0.2",
            "--timeout",
            "500",
        ],
        expected=expected,
        timeout=530,
        decode=False,
    )
    return call(name + "-show", ["show", run, "--json"])


def verify_result(result):
    assert result["complete"] is True, result
    offline = result["stages"]["offline"]
    assert offline["complete"] is True
    assert offline["sample_count"] == 6, offline
    assert offline["anchor_count"] == 170, offline["anchor_count"]
    assert offline["overall"]["prefix-al"]["denominator"] == 158, offline["overall"]
    for name, metric in offline["overall"].items():
        assert metric["denominator"] > 0 and math.isfinite(metric["value"])
        for term in ["numerator", "denominator"]:
            assert math.isclose(
                sum(b["metrics"][name][term] for b in offline["bins"]),
                metric[term],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
    assert 0 <= offline["overall"]["accuracy"]["value"] <= 1
    assert 1 <= offline["overall"]["prefix-al"]["value"] <= 4
    assert all(
        m["value"] is None and m["denominator"] == 0
        for m in offline["bins"][-1]["metrics"].values()
    )
    return offline


def run():
    recipe = create_fixture(ROOT / "fixture")
    call("train-plan", ["train", "-c", str(recipe), "--plan"])
    assert call("empty-list", ["list", "--json"])["runs"] == []
    submitted = call(
        "train-submit",
        [
            "train",
            "-c",
            str(recipe),
            "--name",
            "tiny-dflash",
            "--request-id",
            "e2e-train-1",
            "--json",
        ],
    )
    train = submitted["id"]
    owned.append(train)
    duplicate = call(
        "train-submit-idempotent",
        [
            "train",
            "-c",
            str(recipe),
            "--name",
            "tiny-dflash",
            "--request-id",
            "e2e-train-1",
            "--json",
        ],
    )
    assert duplicate["id"] == train
    with (ROOT / "detached-watch.stdout").open("w") as out:
        observer = subprocess.Popen(
            BASE + ["watch", train, "--jsonl", "--interval", "0.2"],
            stdout=out,
            stderr=subprocess.STDOUT,
        )
        time.sleep(0.5)
        observer.send_signal(signal.SIGINT)
        assert observer.wait(timeout=10) == 0
    assert not (Path(submitted["path"]) / "attempts" / "0001" / "stop.request").exists()
    state = wait("train-reconnect", train)
    assert state["state"] == "succeeded" and state["completion"]["checkpoints"]
    call("train-logs", ["logs", train], decode=False)
    checkpoints = call("train-checkpoints", ["checkpoints", train, "--json"])[
        "checkpoints"
    ]
    assert [c["step"] for c in checkpoints] == [1, 2, 3], checkpoints
    # Compare the same seed/fixture with the existing training command.
    baseline = [
        sys.executable,
        "-m",
        "specforge.cli",
        "train",
        "--config",
        str(recipe),
        "run_id=baseline",
        f'output_dir={ROOT / "baseline"}',
    ]
    with (ROOT / "baseline-worker.log").open("w") as log:
        p = subprocess.run(baseline, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    assert p.returncode == 0, ("baseline", p.returncode)
    old = torch.load(
        ROOT / "baseline" / "baseline-step3" / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    new = torch.load(
        Path(checkpoints[-1]["path"]) / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert old["global_step"] == new["global_step"] == 3
    a, b = old["draft_state_dict"], new["draft_state_dict"]
    assert a.keys() == b.keys()
    maxdiff = max(
        float((a[k].float() - b[k].float()).abs().max())
        for k in a
        if torch.is_tensor(a[k]) and a[k].numel()
    )
    assert maxdiff <= 1e-6, maxdiff
    checks.append({"name": "original-cli-checkpoint-parity", "max_abs_diff": maxdiff})
    resumed = call(
        "resume-step1-submit", ["train", "--resume", train + "@step-1", "--json"]
    )
    assert resumed["id"] == train and resumed["attempt"] == 2, resumed
    resume_state = wait("resume-step1-watch", train)
    assert (
        resume_state["state"] == "succeeded" and resume_state["attempt"] == 2
    ), resume_state
    resumed_checkpoints = call("resume-checkpoints", ["checkpoints", train, "--json"])[
        "checkpoints"
    ]
    resume_step3 = [
        c for c in resumed_checkpoints if c["step"] == 3 and c["attempt"] == "0002"
    ]
    assert len(resume_step3) == 1, resumed_checkpoints
    resumed_payload = torch.load(
        Path(resume_step3[0]["path"]) / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert resumed_payload["global_step"] == 3
    c = resumed_payload["draft_state_dict"]
    assert a.keys() == c.keys()
    resume_diff = max(
        float((a[k].float() - c[k].float()).abs().max())
        for k in a
        if torch.is_tensor(a[k]) and a[k].numel()
    )
    assert resume_diff <= 1e-6, resume_diff
    checks.append(
        {"name": "resume-step1-to3-checkpoint-parity", "max_abs_diff": resume_diff}
    )
    # Every legal anchor is evaluated; bins must pool back into overall.
    config = {
        "recipe": str(recipe),
        "features": str(ROOT / "fixture" / "features"),
        "ctx_origin": "record-start",
        "global_step": 3,
        "total_steps": 3,
        "anchor_chunk_size": 32,
    }
    eval_path = ROOT / "eval.yaml"
    eval_path.write_text(yaml.safe_dump(config))
    args = [
        "eval",
        train + "@latest",
        "-c",
        str(eval_path),
        "--metrics",
        "loss,accuracy,prefix-al",
        "--group-by",
        "ctx-len",
        "--ctx-bins",
        "0,8,16,32,64,inf",
    ]
    call("eval-plan", args + ["--plan"])
    ev = call("eval-submit", args + ["--json"])["id"]
    owned.append(ev)
    assert wait("eval-watch", ev)["state"] == "succeeded"
    result = call("eval-results", ["results", ev, "--json"])
    offline = verify_result(result)
    config["anchor_chunk_size"] = 1
    eval_path2 = ROOT / "eval-chunk1.yaml"
    eval_path2.write_text(yaml.safe_dump(config))
    args2 = [
        "eval",
        train + "@step-3",
        "-c",
        str(eval_path2),
        "--metrics",
        "loss,accuracy,prefix-al",
        "--group-by",
        "ctx-len",
        "--ctx-bins",
        "0,8,16,32,64,inf",
        "--json",
    ]
    ev2 = call("eval-chunk1-submit", args2)["id"]
    owned.append(ev2)
    assert wait("eval-chunk1-watch", ev2)["state"] == "succeeded"
    offline2 = verify_result(call("eval-chunk1-results", ["results", ev2, "--json"]))
    for name in ["loss", "accuracy", "prefix-al"]:
        x, y = offline["overall"][name], offline2["overall"][name]
        assert x["denominator"] == y["denominator"], (name, x, y)
        assert math.isclose(x["value"], y["value"], rel_tol=2e-3, abs_tol=2e-3), (
            name,
            x,
            y,
        )
    checks.append({"name": "anchor-chunk-invariance", "passed": True})
    # A separately owned long run is cancelled; prior completed runs stay intact.
    stopping = call(
        "stop-train-submit",
        [
            "train",
            "-c",
            str(recipe),
            "training.max_steps=100000",
            "training.num_epochs=10000",
            "--json",
        ],
    )["id"]
    owned.append(stopping)
    call("stop-request", ["stop", stopping])
    assert wait("stop-watch", stopping, expected=4)["state"] == "cancelled"
    assert (
        call("original-train-still-succeeded", ["show", train, "--json"])["state"]
        == "succeeded"
    )
    runs = call("final-list", ["list", "--json"])["runs"]
    assert len(runs) == 4, runs
    return {
        "passed": True,
        "checks": checks,
        "runs": [r["id"] for r in runs],
        "original_cli_max_abs_diff": maxdiff,
        "offline_overall": offline["overall"],
        "limitations": [
            "Single CUDA GPU, tiny synthetic DFlash; no remote scheduler adapter or real serving AL validation."
        ],
    }


try:
    result = run()
except BaseException:
    (ROOT / "failure.txt").write_text(traceback.format_exc())
    for run_id in owned:
        subprocess.run(
            BASE + ["stop", run_id], capture_output=True, text=True, timeout=15
        )
    (ROOT / "e2e-result.json").write_text(
        json.dumps(
            {
                "passed": False,
                "checks": checks,
                "runs": owned,
                "error": traceback.format_exc(),
            },
            indent=2,
        )
    )
    raise
else:
    (ROOT / "e2e-result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
