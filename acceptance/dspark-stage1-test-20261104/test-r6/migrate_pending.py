"""Cancel the verified, unstarted r5 and launch an unbound r6 exactly once."""

import json
import os
import select
import signal
import time
import urllib.parse
from pathlib import Path

from common import ROOT, write
from controller import api, single, verify
from platform_ops import connection

SOURCE_ALIAS = "denovo-dspark-test-eval-20261104-r5"
SOURCE_JOB = "job-a5ce2a224dff-20261104153236"


def open_verified_process(record):
    # pidfd pins the process, so an exiting/reused PID cannot signal another task.
    descriptor = os.pidfd_open(record["pid"])
    try:
        base = Path(f"/proc/{record['pid']}")
        fields = (base / "stat").read_text().rsplit(")", 1)[1].split()
        assert fields[0] != "Z" and fields[19] == str(record["start_ticks"])
        command = (base / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        assert command == record["command"], "source command changed"
        assert not select.select([descriptor], [], [], 0)[0], "source already exited"
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def stop_verified_process(descriptor):
    signal.pidfd_send_signal(descriptor, signal.SIGTERM)
    assert select.select([descriptor], [], [], 10)[
        0
    ], "source did not exit; no job cancellation"


def pending_without_worker(source):
    row = single(SOURCE_ALIAS)
    assert row and row["name"] == SOURCE_JOB
    assert row["status"] == "Pending" and row["ready_status"] == "0/1"
    assert row["namespace"] == "qiji" and row["job_creator"] == "linlizhi"
    assert row["user_gpu"] == 8
    overview = connection()(
        "/api/task/get_task_overview?"
        + urllib.parse.urlencode(
            {
                "uuid": SOURCE_JOB,
                "kind": "vcjob",
                "namespace": "qiji",
            }
        )
    )
    assert overview["code"] == 200
    observed = overview["data"]
    assert observed["phase"] == "Pending" and observed["ready_status"] == "0/1"
    (instance,) = observed["instance_info"]["instances"]
    assert (
        not instance["node_name"] and not instance["node_ip"]
    ), "source assigned; do not migrate"
    for name in (
        "phase.json",
        "eval-worker.log",
        "eval-environment.json",
        "result.json",
        "eval-exit.txt",
    ):
        assert not (source / name).exists(), f"source started: {name}; do not migrate"
    assert not any(
        (source / "image-import").iterdir()
    ), "source init started; do not migrate"
    return row


def main():
    verify(require_source_absent=False)
    source = ROOT.parent / "dspark-test-20261104-r5"
    row = pending_without_worker(source)
    record = json.loads((source / "controller-process.json").read_text())
    assert record["pid"] == 5892 and str(record["start_ticks"]) == "2350421198"
    assert record["command"] == f"/usr/bin/python3 -u {source}/controller.py "
    descriptor = open_verified_process(record)
    try:
        platform = connection()
        nodes = platform("/api/k8s/nodes/get_nodes")["nodes"]
        candidates = [
            n
            for n in nodes
            if n["status"] == "Ready"
            and not n.get("unschedulable", False)
            and n.get("node_selector", {}).get("node_shcheduleable") == "true"
            and not n.get("annotation")
            and n.get("gpu_types", {}).get("nvidia.com/gpu") == 8
        ]
        assert candidates, "no allowed GPU node exists"
        write(
            ROOT / "submission-node-preflight.json",
            {
                "time": time.time(),
                "scheduler_selected": True,
                "eligible_by_visible_labels_count": len(candidates),
                "gpu_fragment_counts_not_full_schedulability": True,
                "candidates": [
                    {"name": n["name"], "fragment": n.get("fragment")}
                    for n in candidates
                ],
            },
        )
        with (ROOT / "migration-intent.json").open("x") as stream:
            json.dump(
                {
                    "time": time.time(),
                    "source_job": row,
                    "controller": record,
                    "single_use": True,
                    "pidfd_pinned": True,
                },
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        assert pending_without_worker(source) == row
        stop_verified_process(descriptor)
    finally:
        os.close(descriptor)
    current = pending_without_worker(source)
    write(
        source / "migration-pending-receipt.json",
        {
            "time": time.time(),
            "job": current,
            "no_worker_output": True,
            "reason": "user requested scheduler-selected node after verified node090 memory rejection",
        },
    )
    assert pending_without_worker(source) == current
    deleted = api("delete", alias=SOURCE_ALIAS)
    write(source / "migration-delete.json", deleted)
    assert single(SOURCE_ALIAS) is None
    receipt = {
        "passed": True,
        "time": time.time(),
        "source_job": current,
        "no_worker_output": True,
        "controller_stopped": True,
        "source_alias_absent": True,
        "destination": str(ROOT),
        "delete": deleted,
        "pidfd_pinned": True,
    }
    write(source / "migration-cancellation.json", receipt)
    write(ROOT / "predecessor-cancellation.json", receipt)
    from launch_controller import main as launch

    launch()


if __name__ == "__main__":
    main()
