"""One user-authorized cancellation; preserve partial r9 measurements."""

import json
import os
import select
import signal
import time
from pathlib import Path

from common import ROOT, write
from controller import api

JOB = "job-d1f6f211048d-20261104220148"
ALIAS = "denovo-dspark-test-eval-20261104-r9"
COMMAND = (
    "/usr/bin/python3 -u /gpfs/users/linlizhi/dspark-test-20261104-r9/controller.py "
)


def counts():
    return {
        group: len(list((ROOT / group).glob("test-[0-9][0-9][0-9][0-9].json")))
        for group in ("target-only", "official-mtp", "step25", "step118")
    }


def exact_row():
    rows = api("jobs", alias=ALIAS)
    assert len(rows) == 1 and rows[0]["name"] == JOB
    row = rows[0]
    assert row["namespace"] == "qiji" and row["job_creator"] == "linlizhi"
    return row


def main():
    assert str(ROOT) == "/gpfs/users/linlizhi/dspark-test-20261104-r9"
    before = exact_row()
    rec = json.loads((ROOT / "controller-process.json").read_text())
    assert rec["pid"] == 4064 and str(rec["start_ticks"]) == "2352755370"
    assert rec["command"] == COMMAND
    with (ROOT / "performance-cancellation-intent.json").open("x") as stream:
        json.dump(
            {
                "user_authorization": "重启，然后参考deepinfer分析一下decode 路径低效的地方",
                "job": before,
                "controller": rec,
                "time": time.time(),
                "automatic_retry": False,
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    proc = Path("/proc") / str(rec["pid"])

    def matching():
        try:
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            return (
                fields[19] == str(rec["start_ticks"])
                and fields[0] != "Z"
                and (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
                == COMMAND
            )
        except FileNotFoundError:
            return False

    descriptor = os.pidfd_open(rec["pid"])
    try:
        assert matching(), "controller identity changed; do not signal"
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        select.select([descriptor], [], [], 15)
        assert not matching(), "controller has not exited; do not delete"
    finally:
        os.close(descriptor)
    row = exact_row()
    write(
        ROOT / "performance-terminal-before-delete.json",
        {
            "job": row,
            "controller_stopped": True,
            "phase": json.loads((ROOT / "phase.json").read_text()),
            "test_request_counts": counts(),
            "execution_complete": False,
            "reason": "User requested independent restart with performance settings",
            "time": time.time(),
        },
    )
    deleted = api("delete", alias=ALIAS)
    assert deleted["name"] == JOB and deleted["response"]["success"]
    write(ROOT / "performance-delete.json", deleted)
    for _ in range(15):
        rows = api("jobs", alias=ALIAS)
        if not rows:
            break
        assert len(rows) == 1 and rows[0]["name"] == JOB
        time.sleep(2)
    assert not rows, "deletion not confirmed; do not submit a successor"
    receipt = {
        "passed": True,
        "job": JOB,
        "alias": ALIAS,
        "controller_stopped": True,
        "alias_absent": True,
        "test_request_counts": counts(),
        "execution_complete": False,
        "partial_results_preserved": True,
        "old_execution_manifest_unchanged": True,
        "automatic_retry": False,
        "user_requested_performance_restart": True,
        "time": time.time(),
    }
    write(ROOT / "performance-cancellation.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
