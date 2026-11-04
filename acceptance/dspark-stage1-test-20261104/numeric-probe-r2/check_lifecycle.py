"""Exercise actual child termination, misleading zero exits, and immutable replay guards."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from worker_lifecycle import interrupted_state


def invoke(root, payload):
    driver = "from pathlib import Path;from worker_lifecycle import supervise;import sys,json;sys.exit(supervise(Path(sys.argv[1]),[[sys.executable,'-c',sys.argv[2]]]))"
    return subprocess.Popen(
        [sys.executable, "-c", driver, str(root), payload],
        cwd=Path(__file__).resolve().parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        legacy = root / "legacy"
        legacy.mkdir()
        legacy_script = legacy / "run.sh"
        legacy_script.write_text(
            "#!/bin/bash\ntrap 'echo $? > \"$1/exit.txt\"' EXIT\n"
            'touch "$1/ready"\nsleep 0.3\n'
        )
        process = subprocess.Popen(
            ["bash", str(legacy_script), str(legacy)], start_new_session=True
        )
        deadline = time.monotonic() + 5
        while not (legacy / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (legacy / "ready").exists()
        os.kill(process.pid, signal.SIGTERM)
        assert process.wait(timeout=5) != 0
        assert (legacy / "exit.txt").read_text().strip() == "0"
        complete = root / "complete"
        complete.mkdir()
        result = {
            "probe_complete": True,
            "held_out_requests": 0,
            "request_audit": {"passed": True, "requests": 24},
        }
        payload = f"from pathlib import Path;Path({str(complete / 'result.json')!r}).write_text({json.dumps(result)!r})"
        process = invoke(complete, payload)
        assert process.wait(timeout=5) == 0
        assert (complete / "eval-exit.txt").read_text() == "0\n"
        assert interrupted_state(complete, "Running", True) is None
        original = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in complete.iterdir()
        }
        process = invoke(complete, "raise RuntimeError('replayed payload')")
        assert process.wait(timeout=5) != 0
        assert original == {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in complete.iterdir()
        }
        partial = root / "partial"
        partial.mkdir()
        process = invoke(partial, "pass")
        assert process.wait(timeout=5) != 0
        assert (partial / "eval-exit.txt").read_text() == "1\n"
        assert interrupted_state(partial, "Pending", True)
        untouched = root / "pending"
        untouched.mkdir()
        assert interrupted_state(untouched, "Pending", False) is None
        assert interrupted_state(untouched, "Pending", True)
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            interrupted = root / f"signal-{signum}"
            interrupted.mkdir()
            ready = interrupted / "child-ready"
            # The child deliberately returns0 on TERM/INT. Supervision must retain interruption.
            payload = (
                "import signal,time,sys;from pathlib import Path;"
                "signal.signal(signal.SIGTERM,lambda *args:sys.exit(0));"
                "signal.signal(signal.SIGINT,lambda *args:sys.exit(0));"
                "signal.signal(signal.SIGHUP,lambda *args:sys.exit(0));"
                f"Path({str(ready)!r}).write_text('ready');time.sleep(60)"
            )
            process = invoke(interrupted, payload)
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert ready.exists()
                os.kill(process.pid, signum)
                assert process.wait(timeout=5) == 128 + signum
                assert (
                    interrupted / "eval-exit.txt"
                ).read_text() == f"{128 + signum}\n"
                assert interrupted_state(interrupted, "Running", True)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
        unwind = root / "evaluator-unwind"
        unwind.mkdir()
        ready = unwind / "ready"
        cleanup = unwind / "cleanup"
        payload = (
            "import signal,time\nfrom pathlib import Path\n"
            "from worker_lifecycle import evaluation_signal_handler\n"
            "signal.signal(signal.SIGTERM,evaluation_signal_handler)\n"
            f"Path({str(ready)!r}).write_text('ready')\n"
            "try:\n time.sleep(60)\nfinally:\n"
            f" Path({str(cleanup)!r}).write_text('owned server cleanup reached')\n"
        )
        process = invoke(unwind, payload)
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists()
            os.kill(process.pid, signal.SIGTERM)
            assert process.wait(timeout=5) == 143
            assert cleanup.read_text() == "owned server cleanup reached"
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    print(
        json.dumps(
            {
                "passed": True,
                "actual_term_and_int_zero_exit_children": True,
                "legacy_exit_zero_on_term_reproduced": True,
                "directory_replay_preserves_evidence": True,
                "incomplete_zero_exit_rejected": True,
                "running_to_pending_detected": True,
                "evaluator_signal_unwinds_cleanup": True,
                "hup_preserved": True,
            }
        )
    )


if __name__ == "__main__":
    main()
