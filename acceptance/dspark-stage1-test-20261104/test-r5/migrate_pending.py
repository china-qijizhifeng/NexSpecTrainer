"""Move the still-unstarted owned r4 evaluation to r5 exactly once."""

import json
import os
import signal
import time
from pathlib import Path

from common import ROOT, write
from controller import api, single, verify
from platform_ops import connection


def process(record):
    path = Path(f"/proc/{record['pid']}")
    if not path.exists():
        return False
    fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] == "Z" or fields[19] != str(record["start_ticks"]):
        return False
    command = (path / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    assert command == record["command"], "source PID command changed"
    return True


def main():
    verify(require_source_absent=False)
    source = ROOT.parent / "dspark-test-20261104-r4"
    alias = "denovo-dspark-test-eval-20261104-r4"
    row = single(alias)
    assert row and row["name"] == "job-b44009067f77-20261104112834"
    assert row["status"] == "Pending" and row["ready_status"] == "0/1"
    assert not (source / "phase.json").exists()
    assert not (source / "eval-worker.log").exists()
    record = json.loads((source / "controller-process.json").read_text())
    assert record["pid"] == 56530 and str(record["start_ticks"]) == "2348956988"
    assert process(
        record
    ), "source controller identity is not the expected live process"
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
    with (ROOT / "migration-intent.json").open("x") as stream:
        json.dump(
            {
                "time": time.time(),
                "source_job": row,
                "controller": record,
                "single_use": True,
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.kill(record["pid"], signal.SIGTERM)
    for _ in range(50):
        if not process(record):
            break
        time.sleep(0.1)
    assert not process(record), "source controller still running"
    current = single(alias)
    assert current and current["name"] == row["name"] and current["status"] == "Pending"
    assert (
        not (source / "phase.json").exists()
        and not (source / "eval-worker.log").exists()
    )
    write(
        source / "migration-pending-receipt.json",
        {
            "time": time.time(),
            "job": current,
            "reason": "user prioritized one8GPU run on eligible free node; no worker or test output",
        },
    )
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
    }
    write(source / "migration-cancellation.json", receipt)
    write(ROOT / "predecessor-cancellation.json", receipt)
    from launch_controller import main as launch

    launch()


if __name__ == "__main__":
    main()
