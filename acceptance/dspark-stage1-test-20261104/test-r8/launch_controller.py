"""Single-use launcher, executed once on the SSH host after --check passes."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import ROOT
from controller import verify


def main():
    verify()
    with (ROOT / "controller-launch-intent.json").open("x") as stream:
        json.dump({"time": time.time(), "single_use": True}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    with (ROOT / "controller.log").open("x") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "controller.py")],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    for _ in range(50):
        path = ROOT / "controller-process.json"
        if path.exists():
            record = json.loads(path.read_text())
            assert record["pid"] == process.pid
            stat = (
                Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
            )
            assert stat[19] == str(record["start_ticks"])
            assert (
                Path(f"/proc/{process.pid}/cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode()
                == record["command"]
            )
            print(json.dumps(record), flush=True)
            return
        if process.poll() is not None:
            raise RuntimeError(
                "controller exited before publishing identity; never relaunch"
            )
        time.sleep(0.1)
    raise RuntimeError("controller identity not observed; inspect without relaunching")


if __name__ == "__main__":
    main()
