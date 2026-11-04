"""Cancel only the newly submitted r8-directory job with reused r7 alias."""

import json
import os
import signal
import time
from pathlib import Path

from common import ROOT, write
from controller import api

JOB = "job-d5836fe55d5f-20261104214902"
ALIAS = "denovo-dspark-test-eval-20261104-r7"


def main():
    assert str(ROOT) == "/gpfs/users/linlizhi/dspark-test-20261104-r8"
    with (ROOT / "identity-cancellation-intent.json").open("x") as stream:
        json.dump(
            {
                "job": JOB,
                "alias": ALIAS,
                "reason": "new r8 directory reused old r7 alias",
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    rec = json.loads((ROOT / "controller-process.json").read_text())
    assert rec["pid"] == 53265 and str(rec["start_ticks"]) == "2352679767"
    assert (
        rec["command"]
        == "/usr/bin/python3 -u /gpfs/users/linlizhi/dspark-test-20261104-r8/controller.py "
    )
    proc = Path("/proc") / str(rec["pid"])

    def matches():
        try:
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            return (
                fields[19] == str(rec["start_ticks"])
                and fields[0] != "Z"
                and (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
                == rec["command"]
            )
        except FileNotFoundError:
            return False

    fd = os.pidfd_open(rec["pid"])
    try:
        assert matches()
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        deadline = time.monotonic() + 15
        while matches() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not matches(), "controller still active; do not proceed or relaunch"
    finally:
        os.close(fd)
    rows = api("jobs", alias=ALIAS)
    assert len(rows) == 1 and rows[0]["name"] == JOB
    write(
        ROOT / "identity-terminal-before-delete.json",
        {
            "job": rows[0],
            "controller_stopped": True,
            "worker_started": (ROOT / "worker-started.json").exists(),
            "phase": (
                json.loads((ROOT / "phase.json").read_text())
                if (ROOT / "phase.json").exists()
                else None
            ),
            "time": time.time(),
            "reason": "wrong reused alias in new r8 submission",
        },
    )
    deleted = api("delete", alias=ALIAS)
    assert deleted["name"] == JOB and deleted["response"]["success"]
    write(ROOT / "identity-delete.json", deleted)
    assert api("jobs", alias=ALIAS) == []
    counts = {
        g: len(list((ROOT / g).glob("test-*.json")))
        for g in ("target-only", "official-mtp", "step25", "step118")
    }
    assert not any(counts.values())
    write(
        ROOT / "identity-cancellation.json",
        {
            "passed": True,
            "job": JOB,
            "alias": ALIAS,
            "controller_stopped": True,
            "alias_absent": True,
            "worker_started": (ROOT / "worker-started.json").exists(),
            "test_request_counts": counts,
            "held_out_requests": 0,
            "frozen_r8_files_unchanged": True,
            "automatic_retry": False,
            "reason": "New r8 job used r7 alias because directory-only replacement omitted eval alias naming.",
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "alias_absent": True,
                "held_out_requests": 0,
                "worker_started": (ROOT / "worker-started.json").exists(),
            }
        )
    )


if __name__ == "__main__":
    main()
