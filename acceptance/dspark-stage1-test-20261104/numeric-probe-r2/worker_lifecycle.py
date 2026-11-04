"""Own one worker invocation; preserve interruption and reject replay of its directory."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from common import public_stream


def write(path, value):
    temporary = path.with_name(path.name + ".partial")
    with public_stream(temporary, "w") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def result_complete(root):
    path = root / "result.json"
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    return (
        result.get("probe_complete") is True
        and result.get("held_out_requests") == 0
        and result.get("request_audit", {}).get("passed") is True
        and result.get("request_audit", {}).get("requests") == 24
    )


def supervise(root, commands):
    root = Path(root)
    # This precedes logs/exit files: a platform-created replacement may not overwrite evidence.
    with public_stream(root / "worker-started.json", "x") as stream:
        json.dump(
            {
                "pid": os.getpid(),
                "time": time.time(),
                "job": os.environ.get("JOB_NAME"),
                "node": os.environ.get("NODE_NAME"),
                "single_use": True,
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    interrupted = None
    active = None
    interrupted_at = None
    old_handlers = {}

    def interrupt(signum, frame):
        nonlocal interrupted, interrupted_at
        if interrupted is None:
            interrupted = signum
            interrupted_at = time.monotonic()
            write(
                root / "worker-interrupted.json",
                json.dumps(
                    {
                        "signal": signum,
                        "time": time.time(),
                        "execution_complete": False,
                        "automatic_retry": False,
                    },
                    indent=2,
                )
                + "\n",
            )
            if active is not None and active.poll() is None:
                try:
                    os.killpg(active.pid, signum)
                except ProcessLookupError:
                    pass

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        old_handlers[signum] = signal.signal(signum, interrupt)
    code = 1
    try:
        with public_stream(root / "eval-worker.log", "x") as log:
            for command in commands:
                if interrupted is not None:
                    break
                active = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                # Handle a signal delivered between the pre-spawn check and Popen.
                if interrupted is not None and active.poll() is None:
                    try:
                        os.killpg(active.pid, interrupted)
                    except ProcessLookupError:
                        pass
                while True:
                    try:
                        returned = active.wait(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        if (
                            interrupted_at is not None
                            # Evaluator owns a distinct server group and may need
                            # 90s TERM + 30s KILL cleanup before it can unwind.
                            and time.monotonic() - interrupted_at > 150
                        ):
                            try:
                                os.killpg(active.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            returned = active.wait(timeout=5)
                            break
                active = None
                if interrupted is not None or returned != 0:
                    code = (
                        128 + interrupted
                        if interrupted is not None
                        else (returned if returned > 0 else 128 - returned)
                    )
                    break
            else:
                if not result_complete(root):
                    raise RuntimeError(
                        "worker exited without complete24-request numerical probe audit"
                    )
                code = 0
        if interrupted is not None:
            code = 128 + interrupted
        return code
    except BaseException as error:
        write(
            root / "worker-supervision-failure.json",
            json.dumps(
                {
                    "type": type(error).__name__,
                    "reason": str(error),
                    "automatic_retry": False,
                },
                indent=2,
            )
            + "\n",
        )
        raise
    finally:
        write(
            root / "eval-exit.txt",
            str(128 + interrupted if interrupted is not None else code) + "\n",
        )
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def interrupted_state(root, status, previously_running):
    """Fail closed if a worker exited or a running invocation returns to the queue."""
    root = Path(root)
    if any(
        (root / name).exists()
        for name in (
            "worker-interrupted.json",
            "worker-supervision-failure.json",
            "eval-failure.json",
        )
    ):
        return "worker reported interruption or failure"
    exit_path = root / "eval-exit.txt"
    if exit_path.exists() and not result_complete(root):
        return "worker exited without a complete audited result"
    if previously_running and status == "Pending":
        return "running worker returned to Pending; no directory replay"
    return None


def evaluation_signal_handler(signum, frame):
    # Unwind the evaluator's existing owned-server finally blocks on termination.
    signal.signal(signum, signal.SIG_IGN)
    raise SystemExit(128 + signum)


def main():
    root = Path(__file__).resolve().parent
    from common import verify_execution

    verify_execution()
    commands = [
        [sys.executable, str(root / "copy_cache.py")],
        [
            sys.executable,
            str(root / "with_kernel_cache.py"),
            "--cache-root",
            str(root / "cache"),
            "--namespace",
            "sglang-c6da252c6837-sm90",
            "--",
            sys.executable,
            str(root / "probe.py"),
        ],
    ]
    raise SystemExit(supervise(root, commands))


if __name__ == "__main__":
    main()
