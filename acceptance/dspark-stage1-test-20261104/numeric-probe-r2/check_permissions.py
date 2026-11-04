"""Reproduce old umask failure and verify every public/private writer under both masks."""

import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import common
import probe
import worker_lifecycle


def mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def main():
    previous = (
        Path(__file__).resolve().parent.parent
        / "dspark-aa-numerics-20261104-r1/common.py"
    )
    original_reproduced = False
    if previous.exists():
        spec = importlib.util.spec_from_file_location("failed_probe_common", previous)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old-phase.json"
            saved = os.umask(0o077)
            try:
                old.write(path, {"stage": "fixture"})
            finally:
                os.umask(saved)
            assert mode(path) == 0o600 and not (path.stat().st_mode & stat.S_IROTH)
            original_reproduced = True
    for mask in (0o077, 0o022):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            saved = os.umask(mask)
            try:
                # Common writer covers phase/result/environment/summary/resolved/process metadata.
                for name in (
                    "phase.json",
                    "result.json",
                    "environment.json",
                    "summary.json",
                    "resolved.json",
                    "server-process.json",
                ):
                    common.write(root / name, {"fixture": True})
                    assert mode(root / name) == 0o644
                with patch.object(common, "ROOT", root):
                    common.phase("fixture")
                assert mode(root / "phase.json") == 0o644
                worker_lifecycle.write(root / "eval-exit.txt", "0\n")
                assert mode(root / "eval-exit.txt") == 0o644
                private = root / "private-output.json"
                probe.write_private(private, {"fixture": True})
                assert mode(private) == 0o600
                with probe.private_stream(root / "private.events.jsonl") as stream:
                    stream.write("{}\n")
                assert mode(root / "private.events.jsonl") == 0o600
                try:
                    probe.write_private(private, {"overwrite": True})
                except FileExistsError:
                    pass
                else:
                    raise AssertionError("private evidence overwrite accepted")
                child_root = root / "supervision"
                child_root.mkdir()
                payload = (
                    "from pathlib import Path;from common import write;"
                    f'write(Path({str(child_root / "result.json")!r}),'
                    '{"probe_complete":True,"held_out_requests":0,"request_audit":{"passed":True,"requests":24}})'
                )
                driver = 'from pathlib import Path;from worker_lifecycle import supervise;import sys;sys.exit(supervise(Path(sys.argv[1]),[[sys.executable,"-c",sys.argv[2]]]))'
                result = subprocess.run(
                    [sys.executable, "-c", driver, str(child_root), payload],
                    cwd=Path(__file__).resolve().parent,
                    capture_output=True,
                    timeout=10,
                )
                assert result.returncode == 0
                for name in (
                    "worker-started.json",
                    "eval-worker.log",
                    "eval-exit.txt",
                    "result.json",
                ):
                    assert mode(child_root / name) == 0o644, name
            finally:
                os.umask(saved)
    print(
        json.dumps(
            {
                "passed": True,
                "old_private_control_receipt_reproduced": original_reproduced,
                "public_control_receipts_mode": "0644",
                "private_raw_payload_mode": "0600",
                "parent_umasks_tested": ["077", "022"],
                "real_supervised_child_checked": True,
                "exclusive_private_evidence_guard": True,
                "scope": "POSIX mode bits include other-UID readability for public control metadata; no GPU inference",
            }
        )
    )


if __name__ == "__main__":
    main()
