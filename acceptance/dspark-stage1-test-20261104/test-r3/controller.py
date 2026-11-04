"""Single-use node040 image-import/evaluation; reuse accepted r2 exports."""

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

ALIASES = {"eval": "denovo-dspark-test-eval-20261104-r3"}
PREVIOUS = ROOT.parent / "dspark-test-20261104-r2"
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


def verify():
    verify_execution()
    for step in (25, 118):
        checkpoint_metadata(
            TRAIN / "training" / f"denovo-v41-specforge-stage1-step{step}", step
        )
    (project,) = [p for p in api("partitions")["data"] if p["project_id"] == 32]
    assert project["project_name"] == "WeaverRSI" and project["queue_name"] == "proj-32"
    assert project["enabled"] and project["sync_state"] == "ready"
    assert shutil.disk_usage(ROOT).free > 850_000_000_000 + 40_000_000_000
    manifest = json.loads((ROOT / "inputs-manifest.json").read_text())
    assert (
        manifest["passed"]
        and manifest["test_trajectories"] == 247
        and manifest["test_instances"] == 36
    )
    migration = json.loads((PREVIOUS / "migration-to-r3.json").read_text())
    assert (
        migration["intentional_migration"]
        and migration["original_job"] == "job-3f628626865c-20261104091717"
    )
    assert (
        migration["controller_stopped_verified"]
        and migration["old_job_absent_verified"]
    )
    assert not single("denovo-dspark-test-eval-20261104-r2")
    assert not single("denovo-dspark-test-export-20261104-r2")
    assert not (
        PREVIOUS / "eval-worker.log"
    ).exists(), "old evaluation started; inspect before migration"
    record = json.loads((PREVIOUS / "controller-process.json").read_text())
    stat = Path(f"/proc/{record['pid']}/stat")
    if stat.exists():
        fields = stat.read_text().rsplit(")", 1)[1].split()
        assert fields[19] != str(record["start_ticks"]) or fields[0] == "Z"
    assert json.loads((ROOT / "export-result.json").read_text())["passed"]
    from common import verify_export

    for step in (25, 118):
        verify_export(ROOT, step, read_payload=False)
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
            assert result.get("passed", result.get("execution_complete", False))
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
    if sys.argv[1:] == ["--check"]:
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
