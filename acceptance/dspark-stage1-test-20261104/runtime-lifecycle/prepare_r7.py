"""Prepare a new512Gi attempt; never submit or modify predecessor directories."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

BASE = Path("/Users/shom/Downloads")
OLD = BASE / "dspark-test-20261104-r6"
NEW = BASE / "dspark-test-20261104-r7"
HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / "test-r7"
REFERENCES = {
    "source",
    "export-step25",
    "export-step118",
    "export-step25-verification.json",
    "export-step118-verification.json",
    "test-inputs.jsonl",
    "validation-inputs.jsonl",
}


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def replace(path, old, new):
    value = path.read_text()
    assert old in value, (path.name, old)
    path.write_text(value.replace(old, new))


def prepare():
    NEW.mkdir()
    previous = json.loads((OLD / "execution-sha256.json").read_text())
    names = {
        name for name in previous if "/" not in name and name not in REFERENCES
    } - {"migrate_pending.py", "monitor_supplemental.py"}
    for name in names:
        shutil.copy2(OLD / name, NEW / name)
        if name.endswith((".py", ".json", ".yaml", ".sh", ".md")):
            value = (NEW / name).read_text()
            (NEW / name).write_text(
                value.replace(
                    "dspark-test-20261104-r6", "dspark-test-20261104-r7"
                ).replace("test-eval-20261104-r6", "test-eval-20261104-r7")
            )
    shutil.copytree(
        OLD / "contract-source",
        NEW / "contract-source",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in ("worker_lifecycle.py", "check_lifecycle.py"):
        shutil.copy2(HERE / name, NEW / name)
    shutil.copy2(OLD / "monitor_receipts.py", NEW / "monitor_receipts.py")
    replace(
        NEW / "monitor_receipts.py",
        "dspark-test-20261104-r6",
        "dspark-test-20261104-r7",
    )
    shutil.copy2(
        OLD / "interruption-cancellation.json", NEW / "predecessor-cancellation.json"
    )
    for name in ("eval-job.yaml", "node_contract.py"):
        replace(NEW / name, "768Gi", "512Gi")
    protocol = json.loads((NEW / "protocol.json").read_text())
    protocol["cache_policy"] = protocol["cache_policy"].replace("r6", "r7")
    protocol["resource_policy"] = protocol["resource_policy"].replace("768Gi", "512Gi")
    protocol["migration_from"] = "dspark-test-20261104-r6"
    protocol["memory_amendment"] = {
        "authorization": "那你弄少一点内存",
        "prior_test_requests": 0,
        "requests_and_limits": "512Gi",
        "gpu_cpu_and_generation_unchanged": True,
    }
    protocol["lifecycle_amendment"] = {
        "prior_attempt_interrupted": True,
        "external_cause_confirmed": False,
        "signals_preserved": True,
        "worker_directory_replay_forbidden": True,
        "running_to_pending_stops_with_evidence": True,
        "automatic_retry": False,
    }
    write(NEW / "protocol.json", protocol)
    plan = json.loads((NEW / "resource-plan.json").read_text())
    plan.update(source_attempt="r6", memory="512Gi", cpu=64)
    write(NEW / "resource-plan.json", plan)
    (NEW / "run_eval.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\nTASK_ROOT=/gpfs/users/linlizhi/dspark-test-20261104-r7\n"
        "export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
        "export PYTORCH_ALLOC_CONF=expandable_segments:True\n"
        "export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1 SGLANG_RAGGED_VERIFY_MODE=static\n"
        "unset SGLANG_PLUGINS SPECFORGE_SPEC_CAPTURE SGLANG_SIMULATE_ACC_LEN\n"
        'ulimit -l unlimited\nexec python3 -u "$TASK_ROOT/worker_lifecycle.py"\n'
    )
    replace(
        NEW / "evaluate.py",
        'if __name__ == "__main__":\n    try:',
        'if __name__ == "__main__":\n    from worker_lifecycle import evaluation_signal_handler\n\n'
        "    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):\n"
        "        signal.signal(signum, evaluation_signal_handler)\n    try:",
    )
    controller = NEW / "controller.py"
    value = controller.read_text()
    begin = value.index("    if require_source_absent:")
    end = value.index('    assert json.loads((ROOT / "export-result.json")', begin)
    value = (
        value[:begin]
        + """    previous = ROOT.parent / "dspark-test-20261104-r6"
    receipt = json.loads((previous / "interruption-cancellation.json").read_text())
    assert receipt["alias_absent"] and receipt["controller_stopped"]
    assert receipt["test_requests_completed"] == 0 and not receipt["gpu_aa_complete"]
    assert receipt == json.loads((ROOT / "predecessor-cancellation.json").read_text())
    for old_attempt in (4, 5, 6):
        assert single(f"denovo-dspark-test-eval-20261104-r{old_attempt}") is None
    record = json.loads((previous / "controller-process.json").read_text())
    proc = Path(f"/proc/{record['pid']}/stat")
    if proc.exists():
        fields = proc.read_text().rsplit(")", 1)[1].split()
        assert fields[19] != str(record["start_ticks"]) or fields[0] == "Z"
"""
        + value[end:]
    )
    value = value.replace(
        "    last = None\n    errors = 0",
        "    last = None\n    previously_running = False\n    errors = 0",
    )
    needle = '        if row["status"] in TERMINAL:'
    assert needle in value
    value = value.replace(
        needle,
        """        from worker_lifecycle import interrupted_state

        reason = interrupted_state(ROOT, row["status"], previously_running)
        if reason:
            write(ROOT / "eval-interrupted.json", {
                "job": row, "reason": reason, "time": time.time(),
                "execution_complete": False, "automatic_retry": False,
                "worker_phase": current_phase,
            })
            confirmed = single(alias)
            assert confirmed and confirmed["name"] == identity["name"]
            write(ROOT / "eval-delete.json", api("delete", alias=alias))
            assert single(alias) is None
            raise RuntimeError(reason)
        previously_running = previously_running or row["status"] == "Running"
"""
        + needle,
    )
    controller.write_text(value)
    review = json.loads((NEW / "static-contract-review.json").read_text())
    review.update(
        scope="scheduler-selected512Gi r7; same frozen EOS and988 requests; signal/replay fixes batch-regressed before GPU",
        resource_review=[
            "Single8GPU64CPU512Gi, scheduler-selected allowed node; no new tolerations",
            "Actual8H100 checked in init and regular worker; same fixed original OCI image",
            "r6 interrupted and exact alias absent; controller identity stopped",
            "Supervisor owns child group and preserves TERM/INT/HUP even if child exits0",
            "Evaluator unwinds existing server-group finally cleanup; supervisor grace150s exceeds90+30s server cleanup",
            "worker-started open-x before log creation; Running-to-Pending and incomplete exit cause receipt then exact cancellation",
            "No automatic resubmission; prior execution manifests and receipts unchanged",
        ],
        external_interruption_root_cause="not established",
    )
    write(NEW / "static-contract-review.json", review)
    (NEW / "RUN.md").write_text(
        "# Test r7: scheduler-selected eight H100s,512Gi\n\n"
        "User requested one8GPU group without fixed nodes and lower RAM. Retain64CPU, use512Gi requests/limits. "
        "Only node_shcheduleable=true and existing Weaver tolerations; verify actual8H100 in init and worker. "
        "Keep exact original serving image, source/model hashes, input selection, real EOS/max1024, ten A/A diagnostics and988 requests.\n\n"
        "r6 started but was externally interrupted during A/A, with no completed A/A or test. Its cause remains unconfirmed; "
        "exact job/controller are stopped and alias absent. The next supervisor preserves termination codes, requires a complete audited result "
        "for success and rejects directory replay before opening logs. Evaluator signals unwind owned-server cleanup. "
        "Controller records interruption/exit-without-result or Running-to-Pending, then cancels only its exact job. "
        "Lifecycle/EOS/SSE/scheduling CPU regressions precede this single-use launch; GPU acceptance remains separate. "
        "Do not restart old jobs/controllers or modify predecessor files.\n"
    )
    write(
        NEW / "archive-files.json",
        sorted(
            names
            | {
                "worker_lifecycle.py",
                "check_lifecycle.py",
                "monitor_receipts.py",
                "predecessor-cancellation.json",
            }
        ),
    )
    print(json.dumps({"prepared": str(NEW), "submitted": False}))


def freeze():
    assert not (NEW / "execution-sha256.json").exists()
    previous = json.loads((OLD / "execution-sha256.json").read_text())
    names = json.loads((NEW / "archive-files.json").read_text())
    hashes = {
        name: digest
        for name, digest in previous.items()
        if name.startswith("source/") or name in REFERENCES
    }
    for name in names:
        hashes[name] = hashlib.sha256((NEW / name).read_bytes()).hexdigest()
    write(NEW / "execution-sha256.json", dict(sorted(hashes.items())))
    ARCHIVE.mkdir(exist_ok=True)
    for name in names + ["execution-sha256.json", "archive-files.json"]:
        if name == "r3-eos-reference.json":
            # Captured output token IDs remain in task directories, outside Git.
            continue
        shutil.copy2(NEW / name, ARCHIVE / name)
    write(
        NEW / "source-provenance.json",
        {
            "execution_manifest_entries": len(hashes),
            "execution_source_reused": "r2/source",
            "prior_manifest_sha256": hashlib.sha256(
                (OLD / "execution-sha256.json").read_bytes()
            ).hexdigest(),
            "files_sha256": {name: hashes[name] for name in names},
            "archive_execution_files_byte_identical": True,
            "numerical_inputs_and_variants_unchanged": True,
            "new_worker_memory": "512Gi",
            "fixed_node": False,
            "private_runtime_file_omitted_from_git": ["r3-eos-reference.json"],
        },
    )
    shutil.copy2(NEW / "source-provenance.json", ARCHIVE / "source-provenance.json")
    print(json.dumps({"frozen": len(hashes), "submitted": False}))


if __name__ == "__main__":
    {"prepare": prepare, "freeze": freeze}[sys.argv[1]]()
