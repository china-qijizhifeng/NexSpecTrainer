"""Detached POSIX supervisor. A stop request is a durable file, never a stale PID kill."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from specforge.nsp.store import atomic_json, now, read_json


def supervise(attempt):
    attempt = Path(attempt)
    manifest = read_json(attempt / "manifest.json")
    state = {"state": "starting", "started_at": now(), "supervisor_pid": os.getpid()}
    sequence = 0

    def publish(event, **values):
        nonlocal sequence
        state.update(values, heartbeat_at=now())
        atomic_json(attempt / "status.json", state)
        if event:
            sequence += 1
            with (attempt / "events.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "sequence": sequence,
                            "at": now(),
                            "event": event,
                            **state,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                stream.flush()

    def stop_request(_sig, _frame):
        (attempt / "stop.request").touch()

    signal.signal(signal.SIGTERM, stop_request)
    signal.signal(signal.SIGINT, stop_request)
    child = None
    try:
        publish("starting")
        env = os.environ.copy()
        env["NSP_ATTEMPT_DIR"] = str(attempt)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = (
            str(attempt / "source") + os.pathsep + env.get("PYTHONPATH", "")
        )
        with (attempt / "worker.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(
                manifest["argv"],
                cwd=manifest["cwd"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            publish("running", state="running", worker_pid=child.pid)
            stop_at = None
            offset = 0

            def drain_metrics():
                nonlocal offset
                metrics = attempt / "metrics.jsonl"
                if metrics.exists():
                    with metrics.open() as stream:
                        stream.seek(offset)
                        while True:
                            before = stream.tell()
                            line = stream.readline()
                            if not line or not line.endswith("\n"):
                                offset = before
                                break
                            item = json.loads(line)
                            publish("metrics", latest_metrics=item)

            while child.poll() is None:
                if (attempt / "stop.request").exists() and stop_at is None:
                    stop_at = time.monotonic()
                    publish("stopping", state="stopping")
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                if stop_at is not None and time.monotonic() - stop_at > manifest.get(
                    "stop_grace_seconds", 120
                ):
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                drain_metrics()
                publish(None)
                time.sleep(0.5)
            drain_metrics()
            code = child.wait()
        cancelled = stop_at is not None
        receipt = attempt / "completion.json"
        complete = read_json(receipt) if receipt.is_file() else None
        success = code == 0 and complete is not None
        publish(
            "terminal",
            state="cancelled" if cancelled else "succeeded" if success else "failed",
            finished_at=now(),
            exit_code=code,
            completion=complete,
            error=(
                None
                if success or cancelled
                else "worker failed or did not publish a completion receipt"
            ),
        )
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        publish("terminal", state="failed", finished_at=now(), error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt")
    supervise(parser.parse_args().attempt)


if __name__ == "__main__":
    main()
