"""Cancel only this unstarted probe after discovering private control receipts."""

import json
import os
import signal
import time
from pathlib import Path

from common import ROOT, write
from controller import ALIASES, api, single


def main():
    alias = ALIASES["eval"]
    assert alias == "denovo-dspark-aa-numerics-20261104-r1"
    identity = json.loads((ROOT / "eval-job.json").read_text())
    assert identity["name"] == "job-fe890712eecc-20261104194701"
    row = single(alias)
    assert row and row["name"] == identity["name"] and row["status"] == "Pending"
    assert not (ROOT / "worker-started.json").exists()
    assert not (ROOT / "phase.json").exists()
    reason = "Static inspection: probe umask077 would create root0600 control receipts unreadable by the SSH controller; no numerical inference started."
    with (ROOT / "cancellation-intent.json").open("x") as stream:
        json.dump(
            {"time": time.time(), "reason": reason, "job": row, "single_use": True},
            stream,
            indent=2,
        )
        stream.flush()
        os.fsync(stream.fileno())
    record = json.loads((ROOT / "controller-process.json").read_text())
    stat = Path(f'/proc/{record["pid"]}/stat')
    stopped = not stat.exists()
    if stat.exists():
        fields = stat.read_text().rsplit(")", 1)[1].split()
        if fields[19] == str(record["start_ticks"]) and fields[0] != "Z":
            descriptor = os.pidfd_open(record["pid"])
            try:
                fields = stat.read_text().rsplit(")", 1)[1].split()
                command = (
                    Path(f'/proc/{record["pid"]}/cmdline')
                    .read_bytes()
                    .replace(b"\0", b" ")
                    .decode()
                )
                assert (
                    fields[19] == str(record["start_ticks"])
                    and command == record["command"]
                )
                signal.pidfd_send_signal(descriptor, signal.SIGTERM)
                deadline = time.monotonic() + 30
                while stat.exists() and time.monotonic() < deadline:
                    fields = stat.read_text().rsplit(")", 1)[1].split()
                    if fields[19] != str(record["start_ticks"]) or fields[0] == "Z":
                        break
                    time.sleep(0.1)
            finally:
                os.close(descriptor)
        stopped = (
            not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
        )
    assert stopped
    confirmed = single(alias)
    assert confirmed and confirmed["name"] == identity["name"]
    write(
        ROOT / "cancel-observed.json",
        {
            "job": confirmed,
            "reason": reason,
            "controller_stopped": True,
            "time": time.time(),
        },
    )
    deleted = api("delete", alias=alias)
    write(ROOT / "eval-delete.json", deleted)
    assert single(alias) is None
    result = {
        "cancelled": True,
        "reason": reason,
        "controller_stopped": True,
        "job_alias_absent": True,
        "worker_started": (ROOT / "worker-started.json").exists(),
        "phase_exists": (ROOT / "phase.json").exists(),
        "held_out_requests": 0,
        "automatic_retry": False,
        "time": time.time(),
    }
    write(ROOT / "cancellation.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
