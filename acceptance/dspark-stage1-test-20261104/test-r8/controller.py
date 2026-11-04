"""Single-use EOS-aware evaluation on a scheduler-selected node; reuse r2 exports."""

import fcntl
import io
import json
import os
import shutil
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

from common import ROOT, TRAIN, checkpoint_metadata, verify_execution, write
from northjob_remote import main as northjob

ALIASES = {"eval": "denovo-dspark-test-eval-20261104-r7"}
PREVIOUS = ROOT.parent / "dspark-test-20261104-r3"
TERMINAL = ("Completed", "Succeeded", "Failed", "Aborted", "Terminated")


def api(action, **options):
    out = io.StringIO()
    with redirect_stdout(out):
        northjob(
            {"action": action, "namespace": "qiji", "creator": "linlizhi", **options}
        )
    return json.loads(out.getvalue())


def single(alias):
    rows = api("jobs", alias=alias)
    assert len(rows) <= 1, "ambiguous exact alias"
    return rows[0] if rows else None


def status(stage, **kw):
    value = {
        "stage": stage,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **kw,
    }
    write(ROOT / "pipeline-status.json", value)
    print(json.dumps(value), flush=True)


def verify(require_source_absent=True):
    verify_execution()
    from observation_contract import validate_amendment

    validate_amendment(json.loads((ROOT / "protocol.json").read_text()))
    for step in (25, 118):
        checkpoint_metadata(
            TRAIN / "training" / f"denovo-v41-specforge-stage1-step{step}", step
        )
        from common import verify_export

        verify_export(ROOT, step, read_payload=False)
    (project,) = [p for p in api("partitions")["data"] if p["project_id"] == 32]
    assert project["project_name"] == "WeaverRSI" and project["queue_name"] == "proj-32"
    assert project["enabled"] and project["sync_state"] == "ready"
    assert shutil.disk_usage(ROOT).free > 60_000_000_000
    manifest = json.loads((ROOT / "inputs-manifest.json").read_text())
    assert (
        manifest["passed"]
        and manifest["test_trajectories"] == 247
        and manifest["test_instances"] == 36
    )
    import yaml
    from node_contract import validate_job

    validate_job(yaml.safe_load((ROOT / "eval-job.yaml").read_text()))
    assert json.loads((ROOT / "cpu-regression.json").read_text())["passed"]
    for old_attempt in range(1, 8):
        assert single(f"denovo-dspark-test-eval-20261104-r{old_attempt}") is None
    for number in (1, 2):
        assert single(f"denovo-dspark-aa-numerics-20261104-r{number}") is None
    for directory, job in (
        ("dspark-test-20261104-r7", "job-6b4dd801c01d-20261104173837"),
        ("dspark-aa-numerics-20261104-r2", "job-5c838f4eb38e-20261104200236"),
    ):
        previous = ROOT.parent / directory
        deleted = json.loads((previous / "eval-delete.json").read_text())
        assert deleted["name"] == job and deleted["response"]["success"]
        rec = json.loads((previous / "controller-process.json").read_text())
        proc = Path(f"/proc/{rec['pid']}/stat")
        if proc.exists():
            fields = proc.read_text().rsplit(")", 1)[1].split()
            assert fields[19] != str(rec["start_ticks"]) or fields[0] == "Z"
    assert (
        json.loads((ROOT / "predecessor-cancellation.json").read_text())[
            "prior_held_out_requests"
        ]
        == 0
    )
    for stage, alias in ALIASES.items():
        assert single(alias) is None
        assert not (ROOT / f"{stage}-submit-intent.json").exists()


def run_stage(stage):
    alias = ALIASES[stage]
    assert single(alias) is None
    with (ROOT / f"{stage}-submit-intent.json").open("x") as f:
        json.dump({"alias": alias, "time": time.time(), "single_use": True}, f)
        f.flush()
        os.fsync(f.fileno())
    response = api("submit", yaml=(ROOT / f"{stage}-job.yaml").read_text())
    write(ROOT / f"{stage}-submit-response.json", response)
    assert response.get("code") in (0, 200), response
    identity = None
    for _ in range(30):
        identity = single(alias)
        if identity:
            break
        time.sleep(2)
    assert identity, "submission accepted but not observed; never resubmit"
    write(ROOT / f"{stage}-job.json", identity)
    deadline = time.monotonic() + (4 if stage == "export" else 26) * 3600
    last = None
    previously_running = False
    errors = 0
    while time.monotonic() < deadline:
        try:
            row = single(alias)
        except Exception as exc:
            errors += 1
            write(
                ROOT / "platform-observation-error.json",
                {
                    "consecutive_errors": errors,
                    "type": type(exc).__name__,
                    "time": time.time(),
                },
            )
            if errors >= 5:
                raise
            time.sleep(30)
            continue
        errors = 0
        assert (
            row and row["name"] == identity["name"]
        ), "job identity disappeared/changed"
        current_phase = (
            json.loads((ROOT / "phase.json").read_text())
            if (ROOT / "phase.json").exists()
            else {}
        )
        state = (
            row["status"],
            row.get("ready_status"),
            current_phase.get("stage"),
            current_phase.get("group"),
        )
        if state != last:
            status(stage, job=row, worker_phase=current_phase)
            last = state
        from worker_lifecycle import interrupted_state

        reason = interrupted_state(ROOT, row["status"], previously_running)
        if reason:
            write(
                ROOT / "eval-interrupted.json",
                {
                    "job": row,
                    "reason": reason,
                    "time": time.time(),
                    "execution_complete": False,
                    "automatic_retry": False,
                    "worker_phase": current_phase,
                },
            )
            confirmed = single(alias)
            assert confirmed and confirmed["name"] == identity["name"]
            write(ROOT / "eval-delete.json", api("delete", alias=alias))
            assert single(alias) is None
            raise RuntimeError(reason)
        previously_running = previously_running or row["status"] == "Running"
        if row["status"] in TERMINAL:
            write(ROOT / f"{stage}-terminal.json", row)
            confirmed = single(alias)
            assert (
                confirmed
                and confirmed["name"] == identity["name"]
                and confirmed["status"] in TERMINAL
            )
            write(ROOT / f"{stage}-delete.json", api("delete", alias=alias))
            assert single(alias) is None
            assert row["status"] in (
                "Completed",
                "Succeeded",
            ), f"{stage} failed; no retry"
            assert (ROOT / f"{stage}-exit.txt").read_text().strip() == "0"
            result = json.loads(
                (
                    ROOT
                    / ("export-result.json" if stage == "export" else "result.json")
                ).read_text()
            )
            from worker_lifecycle import result_complete

            assert result_complete(ROOT)
            return
        assert row["status"] in ("Pending", "Running"), row
        time.sleep(30)
    # Bound the lifetime of only this attempt, after saving timeout identity evidence.
    row = single(alias)
    write(ROOT / f"{stage}-timeout.json", {"job": row, "time": time.time()})
    assert row and row["name"] == identity["name"]
    write(ROOT / f"{stage}-delete.json", api("delete", alias=alias))
    assert single(alias) is None
    raise TimeoutError(f"{stage} exceeded bounded lifetime; no retry")


def run():
    verify()
    status("verified")
    run_stage("eval")
    status(
        "complete",
        result="result.json",
        all_aliases_absent=all(single(a) is None for a in ALIASES.values()),
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--prepare-check"]:
        verify(require_source_absent=False)
        print(json.dumps({"passed": True, "read_only": True}))
    elif sys.argv[1:] == ["--check"]:
        verify()
        print(json.dumps({"passed": True, "read_only": True}))
    elif not sys.argv[1:]:
        with (ROOT / "controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (ROOT / "controller-started.json").open("x") as f:
                json.dump(
                    {"pid": os.getpid(), "time": time.time(), "single_use": True}, f
                )
                f.flush()
                os.fsync(f.fileno())
            write(
                ROOT / "controller-process.json",
                {
                    "pid": os.getpid(),
                    "start_ticks": Path(f"/proc/{os.getpid()}/stat")
                    .read_text()
                    .rsplit(")", 1)[1]
                    .split()[19],
                    "command": Path(f"/proc/{os.getpid()}/cmdline")
                    .read_bytes()
                    .replace(b"\0", b" ")
                    .decode(),
                    "root": str(ROOT),
                },
            )
            try:
                run()
            except Exception as exc:
                status(
                    "failed",
                    error_type=type(exc).__name__,
                    reason=str(exc),
                    automatic_retry=False,
                )
                raise SystemExit(1) from None
    else:
        raise SystemExit("use --check or no arguments")
