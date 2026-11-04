"""Continue only the audited r4 SIGTERM exit; never rerun the migration launcher."""

import json
import time
from pathlib import Path

from common import ROOT, write
from controller import api, single, verify
from platform_ops import connection


def main():
    verify(require_source_absent=False)
    source = ROOT.parent / "dspark-test-20261104-r4"
    alias = "denovo-dspark-test-eval-20261104-r4"
    intent = json.loads((ROOT / "migration-intent.json").read_text())
    record = json.loads((source / "controller-process.json").read_text())
    assert intent["controller"] == record
    assert record["pid"] == 56530 and str(record["start_ticks"]) == "2348956988"
    assert not Path(f"/proc/{record['pid']}").exists(), "old process must be absent"
    assert not any(
        (ROOT / name).exists()
        for name in (
            "controller-launch-intent.json",
            "controller-started.json",
            "eval-submit-intent.json",
            "predecessor-cancellation.json",
        )
    )
    assert not (source / "phase.json").exists()
    assert not (source / "eval-worker.log").exists()
    row = single(alias)
    assert row == intent["source_job"]
    assert row["name"] == "job-b44009067f77-20261104112834"
    assert row["status"] == "Pending" and row["ready_status"] == "0/1"
    platform = connection()
    (node,) = [
        n
        for n in platform("/api/k8s/nodes/get_nodes")["nodes"]
        if n["name"] == "node090"
    ]
    assert node["status"] == "Ready" and not node.get("unschedulable", False)
    assert node["node_selector"]["node_shcheduleable"] == "true"
    assert not node.get("annotation") and node.get("fragment") == 8
    assert node["gpu_types"]["nvidia.com/gpu"] == 8
    write(ROOT / "submission-node-preflight.json", node)
    with (ROOT / "migration-exit-recovery-intent.json").open("x") as stream:
        json.dump(
            {
                "time": time.time(),
                "source_process_absent": True,
                "source_job": row,
                "reason": "SIGTERM exit raced stat/cmdline reads; original intent retained",
                "source_worker_never_started": True,
                "single_use": True,
            },
            stream,
        )
        stream.flush()
    write(
        source / "migration-pending-receipt.json",
        {"time": time.time(), "job": row, "no_worker_output": True},
    )
    current = single(alias)
    assert current == row and not (source / "eval-worker.log").exists()
    deleted = api("delete", alias=alias)
    write(source / "migration-delete.json", deleted)
    assert single(alias) is None
    receipt = {
        "passed": True,
        "time": time.time(),
        "source_job": row,
        "no_worker_output": True,
        "controller_stopped": True,
        "source_alias_absent": True,
        "destination": str(ROOT),
        "delete": deleted,
        "recovered_exit_observation_race": True,
    }
    write(source / "migration-cancellation.json", receipt)
    write(ROOT / "predecessor-cancellation.json", receipt)
    from launch_controller import main as launch

    launch()


if __name__ == "__main__":
    main()
