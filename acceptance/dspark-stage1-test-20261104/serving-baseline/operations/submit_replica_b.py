"""Submit the frozen second replica once, after actual first-worker acceptance."""

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

from execution_contract import verify_execution
from platform_ops import connection, write

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "replica-b"
ALIAS = "dspark-v41-baseline-b-20261104-r1"


def check(api):
    verify_execution(ROOT)
    acceptance = json.loads((ROOT / "replica-a/api-acceptance.json").read_text())
    assert acceptance["passed"] and acceptance["worker"] == "replica-a"
    assert not acceptance["lossless_greedy_verified"]
    phase = json.loads((ROOT / "replica-a/phase.json").read_text())
    assert phase["stage"] == "worker_ready"
    rows = api("/api/get_jobs")["data"]
    first = json.loads((ROOT / "replica-a/job.json").read_text())
    current = [r for r in rows if r.get("user_def_name") == first["user_def_name"]]
    assert len(current) == 1
    assert all(current[0][k] == first[k] for k in ("name", "namespace", "job_creator"))
    assert current[0]["status"] == "Running"
    assert not any(r.get("user_def_name") == ALIAS for r in rows)
    assert not (OUT / "submit-intent.json").exists()
    projects = api("/api/partitions")["data"]
    (project,) = [r for r in projects if r["project_id"] == 32]
    assert project["project_name"] == "WeaverRSI" and project["queue_name"] == "proj-32"
    assert project["enabled"] and project["sync_state"] == "ready"
    nodes = api("/api/k8s/nodes/get_nodes")["nodes"]
    (node,) = [r for r in nodes if r["name"] == "node207"]
    assert node["status"] == "Ready" and not node.get("unschedulable", False)
    assert node["node_selector"]["node_shcheduleable"] == "true"
    assert node["node_selector"]["kubernetes.io/hostname"] == "node207"
    assert not node.get("annotation"), "node reservation annotation requires review"
    assert node["gpu_types"]["nvidia.com/gpu"] == 8
    assert shutil.disk_usage(ROOT).free > 850_000_000_000 + 140_000_000_000
    result = {
        "passed": True,
        "time": time.time(),
        "project": project,
        "node": node,
        "available_gpu_slots_observed": node.get("fragment"),
        "actual_card_type_checked_in_init": True,
        "first_acceptance_sha256": hashlib.sha256(
            (ROOT / "replica-a/api-acceptance.json").read_bytes()
        ).hexdigest(),
    }
    write(OUT / "submission-preflight.json", result)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    api = connection()
    checked = check(api)
    if sys.argv[1:] == ["--check"]:
        print(
            json.dumps(
                {
                    "passed": True,
                    "node": "node207",
                    "fragment": checked["node"].get("fragment"),
                }
            )
        )
        return
    assert not sys.argv[1:]
    yaml = (ROOT / "replica-b-job.yaml").read_text()
    with (OUT / "submit-intent.json").open("x") as stream:
        json.dump(
            {
                "alias": ALIAS,
                "time": time.time(),
                "single_use": True,
                "yaml_sha256": hashlib.sha256(yaml.encode()).hexdigest(),
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    response = api("/api/task/create_job_by_yaml", "POST", {"code": yaml})
    write(OUT / "submit-response.json", response)
    assert response["code"] in (0, 200), "submission uncertain or rejected; never retry"
    for _ in range(30):
        rows = [
            r for r in api("/api/get_jobs")["data"] if r.get("user_def_name") == ALIAS
        ]
        assert len(rows) <= 1
        if rows:
            row = rows[0]
            assert (
                row["namespace"] == "qiji"
                and row["job_creator"] == "linlizhi"
                and row["user_gpu"] == 8
            )
            identity = {
                k: row.get(k)
                for k in (
                    "name",
                    "user_def_name",
                    "status",
                    "ready_status",
                    "namespace",
                    "job_creator",
                    "user_gpu",
                )
            }
            write(OUT / "job.json", identity)
            print(json.dumps(identity))
            return
        time.sleep(2)
    raise RuntimeError("submission accepted but identity not observed; never retry")


if __name__ == "__main__":
    main()
