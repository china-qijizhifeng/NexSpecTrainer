"""Delete a bounded plan of superseded huangyuqi checkpoints through Weaver."""

import argparse
import collections
import datetime as dt
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OWNER = "local-314a44be3bed7a757700389ecc10233f"
REMOTE = r'''
import json, shutil, sys
from pathlib import Path
for name in ["ZhiFm-FTJb69v89x", "4_z9iMKUlIpOxg25", "zRx6mCeBcoUnUOwEYYYXc"]:
    sys.path.insert(0, "/gpfs/users/linlizhi/.cache/uv/archive-v0/" + name)
import psycopg
from psycopg.rows import dict_row
root = Path("/gpfs/users/linlizhi/weaver-public-rsi-20261028/control")
env = json.loads((root / "secrets/environment.json").read_text())
with psycopg.connect(env["WEAVER_DATABASE_URL"], options="-c default_transaction_read_only=on", row_factory=dict_row) as db:
    profile = db.execute("SELECT user_id,username FROM user_profiles WHERE user_id=%s", (OWNER,)).fetchone()
    assert profile["username"] == "huangyuqi"
    rows = db.execute("""
        SELECT c.id,c.model_id,c.principal_id,c.name,c.kind,c.status,c.generation,
               c.checkpoint_step,c.purpose,c.created_at,c.manifest_digest,c.output_prefix,
               c.deletion_state,c.desired_delete,
               (SELECT COALESCE(sum((f->>'size')::bigint),0)
                FROM jsonb_array_elements(c.manifest->'files') f) AS size_bytes,
               COALESCE((SELECT jsonb_agg(jsonb_build_object('kind',owner_kind,'id',owner_id))
                         FROM checkpoint_references WHERE checkpoint_id=c.id),'[]'::jsonb) AS refs,
               (SELECT count(*) FROM model_runs r WHERE r.source_checkpoint_id=c.id
                AND r.lifecycle<>'sealed') AS active_sources,
               (SELECT count(*) FROM model_checkpoints child WHERE child.source_id=c.id
                AND child.deletion_state<>'deleted') AS derived_sources
        FROM model_checkpoints c WHERE c.principal_id=%s AND c.deletion_state<>'deleted'
        ORDER BY c.created_at DESC
    """, (OWNER,)).fetchall()
print(json.dumps({"profile":profile,"free_bytes":shutil.disk_usage(root).free,"rows":rows},default=str))
'''


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + "\n")


def snapshot():
    result = subprocess.run(
        [
            "ssh",
            "linlizhi-bp-dev",
            "PYTHONDONTWRITEBYTECODE=1 /gpfs/users/linlizhi/miniconda3/bin/python -",
        ],
        input="OWNER=" + repr(OWNER) + "\n" + REMOTE,
        text=True,
        capture_output=True,
        check=True,
        timeout=90,
    )
    data = json.loads(result.stdout)
    for row in data["rows"]:
        for key in (
            "generation",
            "checkpoint_step",
            "size_bytes",
            "active_sources",
            "derived_sources",
        ):
            row[key] = int(row[key])
    return data


def api(auth, method, path):
    req = urllib.request.Request(
        auth["base_url"].rstrip("/") + path,
        method=method,
        headers={"X-WEAVER-API-KEY": auth["api_key"]},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.status, json.load(response)


def eligible(row):
    return (
        row["principal_id"] == OWNER
        and row["kind"] == "trainer"
        and row["purpose"] == "user"
        and row["status"] == "ready"
        and row["deletion_state"] == "live"
        and not row["desired_delete"]
        and not row["refs"]
        and not row["active_sources"]
        and not row["derived_sources"]
    )


def plan():
    data = snapshot()
    save("database-before.json", data)
    groups = collections.defaultdict(list)
    for row in data["rows"]:
        if row["status"] == "ready" and row["kind"] == "trainer":
            groups[(row["model_id"], row["generation"])].append(row)
    protected = {r["id"] for group in groups.values() for r in group[:2]}
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
    candidates = [
        r
        for r in data["rows"]
        if eligible(r)
        and r["id"] not in protected
        and dt.datetime.fromisoformat(r["created_at"]) < cutoff
    ]
    selected = []
    for row in reversed(candidates):
        selected.append(row)
        if sum(r["size_bytes"] for r in selected) >= 1_000_000_000_000:
            break
    value = {
        "owner": OWNER,
        "keep_latest_ready_per_model_generation": 2,
        "minimum_age_hours": 1,
        "api_only": True,
        "selected": selected,
        "protected_latest_ids": sorted(protected),
        "logical_bytes": sum(r["size_bytes"] for r in selected),
    }
    save("deletion-plan.json", value)
    print(json.dumps(value, indent=2))


def apply():
    plan_data = json.loads((ROOT / "deletion-plan.json").read_text())
    assert plan_data["owner"] == OWNER and 0 < len(plan_data["selected"]) <= 12
    assert not (
        ROOT / "apply-intent.json"
    ).exists(), "single-use; inspect existing receipts"
    auth = json.JSONDecoder().raw_decode(
        Path(
            "/Users/shom/.config/weaver/public-rsi-20261028/huangyuqi-user-key.json"
        ).read_text()
    )[0]
    _, identity = api(auth, "GET", "/api/v1/users/me")
    assert identity["user_id"] == OWNER and identity["username"] == "huangyuqi"
    fresh = snapshot()
    by_id = {r["id"]: r for r in fresh["rows"]}
    for row in plan_data["selected"]:
        current = by_id[row["id"]]
        assert (
            eligible(current) and current["manifest_digest"] == row["manifest_digest"]
        )
    save("database-pre-delete.json", fresh)
    with (ROOT / "apply-intent.json").open("x") as stream:
        json.dump(
            {
                "ids": [r["id"] for r in plan_data["selected"]],
                "authorization": "user explicitly requested old Weaver huangyuqi ckpt cleanup",
            },
            stream,
        )
    receipts = []
    for row in plan_data["selected"]:
        status, result = api(auth, "DELETE", "/api/v1/checkpoints/" + row["id"])
        receipts.append(
            {
                "id": row["id"],
                "name": row["name"],
                "http_status": status,
                "deletion_state": result.get("deletion_state"),
                "cleanup_pending": result.get("cleanup_pending"),
            }
        )
        save("delete-receipts.json", receipts)
        assert status == 202 and result["id"] == row["id"]
        assert result["deletion_state"] == "deleted" and not result["cleanup_pending"]
        print(json.dumps(receipts[-1]), flush=True)
    after = snapshot()
    assert not set(r["id"] for r in after["rows"]) & set(
        r["id"] for r in plan_data["selected"]
    )
    assert set(plan_data["protected_latest_ids"]) <= set(r["id"] for r in after["rows"])
    save("database-after.json", after)
    save(
        "cleanup-result.json",
        {
            "passed": True,
            "deleted_count": len(receipts),
            "logical_bytes_removed": plan_data["logical_bytes"],
            "free_before": fresh["free_bytes"],
            "free_after": after["free_bytes"],
            "latest_preserved": True,
            "api_managed_deletion": True,
        },
    )


def bootstrap():
    """Recover database write space using one old, superseded sealed-run save."""
    checkpoint = "7c4b0267685852a11f0124841c7efea6"
    model = "9c81d3f9a6d8564c389ec6e9734de25c"
    auth = json.JSONDecoder().raw_decode(
        Path(
            "/Users/shom/.config/weaver/public-rsi-20261028/huangyuqi-user-key.json"
        ).read_text()
    )[0]
    _, user = api(auth, "GET", "/api/v1/users/me")
    assert user["user_id"] == OWNER and user["username"] == "huangyuqi"
    _, state = api(auth, "GET", "/api/v1/models/" + model)
    assert state["status"] == "sealed" and state["desired_delete"]
    _, checkpoints = api(
        auth, "GET", "/api/v1/models/" + model + "/checkpoints?limit=200"
    )
    ready = [
        c
        for c in checkpoints["items"]
        if c["status"] == "completed"
        and c["deletion_state"] == "live"
        and not c["desired_delete"]
    ]
    assert len(ready) >= 3 and checkpoints["pagination"]["total_count"] == len(
        checkpoints["items"]
    )
    ordered = sorted(ready, key=lambda c: c["created_at"], reverse=True)
    row = next(c for c in ready if c["id"] == checkpoint)
    assert checkpoint not in [c["id"] for c in ordered[:2]]
    assert row["name"] == "output-epoch-01" and row["created_at"].startswith(
        "2026-10-29"
    )
    save(
        "bootstrap-before.json",
        {
            "model": state,
            "checkpoint": row,
            "preserved_ids": [c["id"] for c in ordered[:2]],
            "reference_guard": "Weaver Delete/RestoreDeletion checks live references under transaction",
        },
    )
    with (ROOT / "bootstrap-intent.json").open("x") as stream:
        json.dump({"id": checkpoint, "single_use": True, "api_only": True}, stream)
    status, result = api(auth, "DELETE", "/api/v1/checkpoints/" + checkpoint)
    save("bootstrap-delete.json", {"http_status": status, "result": result})
    assert status == 202 and result["deletion_state"] == "deleted"
    print(
        json.dumps(
            {"id": checkpoint, "deleted": True, "logical_bytes": row["size_bytes"]}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "apply", "bootstrap"])
    args = parser.parse_args()
    {"plan": plan, "apply": apply, "bootstrap": bootstrap}[args.action]()
