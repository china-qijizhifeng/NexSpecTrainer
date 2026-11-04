"""Stop only verified r2 controller and its still-pending exact job, once."""

import json
import os
import signal
import time
from pathlib import Path

from common import ROOT, write
from controller import api, single


def run():
    previous = ROOT.parent / "dspark-test-20261104-r2"
    record = json.loads((previous / "controller-process.json").read_text())
    expected_command = "python3 -u " + str(previous / "controller.py") + " "
    assert record["pid"] == 25507 and record["command"] == expected_command
    proc = Path("/proc") / str(record["pid"])
    assert proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19] == str(
        record["start_ticks"]
    )
    assert (
        proc.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode()
        == record["command"]
    )
    alias = "denovo-dspark-test-eval-20261104-r2"
    row = single(alias)
    assert (
        row
        and row["name"] == "job-3f628626865c-20261104091717"
        and row["status"] == "Pending"
    )
    assert (
        not (previous / "eval-worker.log").exists()
        and not (previous / "eval-environment.json").exists()
    )
    with (ROOT / "migration-intent.json").open("x") as f:
        json.dump(
            {
                "time": time.time(),
                "controller": record,
                "job": row,
                "reason": "user requested image injection on another node",
            },
            f,
        )
        f.flush()
        os.fsync(f.fileno())
    fd = os.pidfd_open(record["pid"])
    try:
        # Recheck after opening the identity-bound process handle, before signal.
        assert proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19] == str(
            record["start_ticks"]
        )
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)
    stopped = False
    for _ in range(20):
        if not proc.joinpath("stat").exists():
            stopped = True
            break
        stat = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()
        if stat[19] != str(record["start_ticks"]) or stat[0] == "Z":
            stopped = True
            break
        time.sleep(0.5)
    assert stopped, "old controller did not exit; inspect, never submit"
    row = single(alias)
    assert (
        row
        and row["name"] == "job-3f628626865c-20261104091717"
        and row["status"] == "Pending"
    )
    write(previous / "migration-pending-job.json", row)
    response = api("delete", alias=alias)
    write(previous / "migration-delete.json", response)
    assert single(alias) is None
    result = {
        "intentional_migration": True,
        "original_job": row["name"],
        "original_alias": alias,
        "controller_stopped_verified": True,
        "old_job_absent_verified": True,
        "old_evaluation_started": False,
        "new_root": str(ROOT),
        "new_node": "node040",
        "time": time.time(),
        "reason": "user requested same-digest image injection on a free node; preserve exports and frozen test",
    }
    write(previous / "migration-to-r3.json", result)
    write(ROOT / "migration-from-r2.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    run()
