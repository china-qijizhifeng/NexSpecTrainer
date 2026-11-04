"""Check submitted scheduling contract and Linux pidfd exit behavior with owned children."""

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from node_contract import validate_job, validate_node


def rejected(call):
    try:
        call()
    except AssertionError:
        return
    raise AssertionError("unsafe configuration accepted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--linux-process-check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    original = yaml.safe_load((root / "eval-job.yaml").read_text())
    validate_job(original)
    for hostname in ("node090", "node205", "new-gpu-node.example"):
        validate_node(hostname)
    rejected(lambda: validate_node(""))
    for field, value in (
        ("nodeName", "node090"),
        (
            "affinity",
            {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {}}},
        ),
        (
            "nodeSelector",
            {"node_shcheduleable": "true", "kubernetes.io/hostname": "node090"},
        ),
        ("tolerations", [{"operator": "Exists"}]),
    ):
        altered = copy.deepcopy(original)
        altered["spec"]["tasks"][0]["template"]["spec"][field] = value
        rejected(lambda: validate_job(altered))
    if args.linux_process_check:
        from migrate_pending import open_verified_process, stop_verified_process

        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import time; print('ready', flush=True); time.sleep(30)",
            ],
            start_new_session=True,
            stdout=subprocess.PIPE,
        )
        try:
            # Popen can return before exec; observe identity after the child is ready.
            assert process.stdout.readline() == b"ready\n"
            base = Path(f"/proc/{process.pid}")
            record = {
                "pid": process.pid,
                "start_ticks": (base / "stat")
                .read_text()
                .rsplit(")", 1)[1]
                .split()[19],
                "command": (base / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(),
            }
            rejected(lambda: open_verified_process({**record, "start_ticks": "-1"}))
            rejected(
                lambda: open_verified_process(
                    {**record, "command": "different command "}
                )
            )
            assert process.poll() is None
            descriptor = open_verified_process(record)
            try:
                stop_verified_process(descriptor)
                assert process.wait(timeout=2) == -15
            finally:
                os.close(descriptor)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            process.stdout.close()
    print(
        json.dumps(
            {
                "passed": True,
                "scheduler_selected": True,
                "forbidden_placement_and_tolerations_rejected": True,
                "linux_pidfd_owned_child": args.linux_process_check,
            }
        )
    )


if __name__ == "__main__":
    main()
