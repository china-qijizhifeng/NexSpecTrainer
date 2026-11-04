"""Prepare a new observational benchmark; never mutate or launch prior attempts."""

import json
from pathlib import Path

OLD = Path("/Users/shom/Downloads/dspark-test-20261104-r7")
ROOT = OLD.with_name("dspark-test-20261104-r8")
PROBE = OLD.with_name("dspark-aa-numerics-20261104-r2")


def main():
    ROOT.mkdir()
    (ROOT / "image-import").mkdir()
    names = [
        "evaluate.py",
        "controller.py",
        "launch_controller.py",
        "monitor.py",
        "monitor_receipts.py",
        "node_contract.py",
        "platform_ops.py",
        "northjob_remote.py",
        "copy_cache.py",
        "ctr_contract.py",
        "import_worker_image.py",
        "with_kernel_cache.py",
        "execution_contract.py",
        "check_runner.py",
        "check_eos.py",
        "check_lifecycle.py",
        "check_scheduling.py",
        "eval-job.yaml",
        "run_eval.sh",
        "serving-source-pins.json",
        "inputs-manifest.json",
        "cached-image-evidence.json",
        "export-result.json",
    ]
    for name in names:
        text = (
            (OLD / name)
            .read_text()
            .replace("dspark-test-20261104-r7", "dspark-test-20261104-r8")
        )
        (ROOT / name).write_text(text)
    (ROOT / "common.py").write_bytes((PROBE / "common.py").read_bytes())
    lifecycle = (PROBE / "worker_lifecycle.py").read_text()
    start = lifecycle.index("def result_complete(root):")
    end = lifecycle.index("\n\ndef supervise", start)
    formal = (OLD / "worker_lifecycle.py").read_text()
    fstart = formal.index("def result_complete(root):")
    fend = formal.index("\n\ndef supervise", fstart)
    lifecycle = lifecycle[:start] + formal[fstart:fend] + lifecycle[end:]
    lifecycle = lifecycle.replace(
        "complete24-request numerical probe audit",
        "complete988-request benchmark audit",
    )
    lifecycle = lifecycle.replace('str(root / "probe.py")', 'str(root / "evaluate.py")')
    (ROOT / "worker_lifecycle.py").write_text(lifecycle)
    protocol = json.loads((OLD / "protocol.json").read_text())
    protocol["correctness"] = (
        "Observe all10 full target-only A/A pairs and frozen61 reference without using numerical equality as a run gate. Record differences, retain actual EOS/HTTP/count/finite assertions, then run all4x247 fixed requests. No lossless claim."
    )
    protocol["observational_amendment"] = {
        "authorization": "算了先容忍这种差距，看看分数和接受绿坝",
        "clarification": "先看当前测评指标",
        "prior_held_out_requests": 0,
        "allow_aa_output_mismatch": True,
        "allow_historical_reference_mismatch": True,
        "numerical_root_cause_fixed": False,
        "lossless_claim_allowed": False,
        "retain_eos_http_counts_finite_input_identity_guards": True,
        "metrics": [
            "observed throughput",
            "native draft accept rate",
            "verify weighted accept length",
            "TTFT",
            "TPOT",
            "output equality",
        ],
        "automation_remains_paused": True,
    }
    protocol["cache_policy"] = protocol["cache_policy"].replace("r7", "r8")
    protocol["migration_from"] = "r7 terminal failure; independent r8 observational run"
    (ROOT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n"
    )
    plan = json.loads((OLD / "resource-plan.json").read_text())
    plan["source_attempt"] = "r7; reuse frozen inputs/source/r2 exports"
    (ROOT / "resource-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (ROOT / "prepare_r8.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {"prepared": str(ROOT), "launched": False, "automation_resumed": False}
        )
    )


if __name__ == "__main__":
    main()
