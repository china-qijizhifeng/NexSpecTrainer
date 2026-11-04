"""Prepare the user-authorized single-eight-H100 scheduling migration, without submitting."""

import json
import shutil
from pathlib import Path

import yaml

OLD = Path("/Users/shom/Downloads/dspark-test-20261104-r4")
NEW = OLD.with_name("dspark-test-20261104-r5")
SERVING = OLD.with_name("dspark-v41-northgate-20261104-r1")
REMOTE = "/gpfs/users/linlizhi/dspark-test-20261104-r5"


def replace(path, before, after):
    value = path.read_text()
    assert before in value, (path, before)
    path.write_text(value.replace(before, after))


def main():
    NEW.mkdir()
    for name in json.loads((OLD / "archive-files.json").read_text()) + [
        "inputs-manifest.json",
        "export-result.json",
        "r3-eos-reference.json",
        "cached-image-evidence.json",
        "execution-sha256.json",
        "archive-files.json",
    ]:
        shutil.copy2(OLD / name, NEW / name)
    for name in ["ctr_contract.py", "import_worker_image.py"]:
        shutil.copy2(SERVING / name, NEW / name)
    (NEW / "execution_contract.py").write_text(
        "def verify_execution(root):\n    from common import verify_execution as check\n    check()\n"
    )
    replace(NEW / "run_eval.sh", "dspark-test-20261104-r4", "dspark-test-20261104-r5")
    replace(NEW / "monitor.py", "dspark-test-20261104-r4", "dspark-test-20261104-r5")
    replace(
        NEW / "controller.py",
        "denovo-dspark-test-eval-20261104-r4",
        "denovo-dspark-test-eval-20261104-r5",
    )
    replace(
        NEW / "controller.py",
        "850_000_000_000 + 40_000_000_000",
        "50_000_000_000 + 10_000_000_000",
    )
    value = (NEW / "controller.py").read_text()
    start = value.index("    imported = json.loads(")
    end = value.index('    assert json.loads((ROOT / "export-result.json")', start)
    value = (
        value[:start]
        + """    plan = json.loads((ROOT / "resource-plan.json").read_text())
    assert plan["node"] == "node090" and plan["gpus"] == 8
    if require_source_absent:
        previous = ROOT.parent / "dspark-test-20261104-r4"
        receipt = json.loads((previous / "migration-cancellation.json").read_text())
        assert receipt["passed"] and receipt["no_worker_output"]
        assert single("denovo-dspark-test-eval-20261104-r4") is None
"""
        + value[end:]
    )
    value = value.replace("def verify():", "def verify(require_source_absent=True):")
    value = value.replace(
        '    if sys.argv[1:] == ["--check"]:',
        '    if sys.argv[1:] == ["--prepare-check"]:\n        verify(require_source_absent=False)\n        print(json.dumps({"passed": True, "read_only": True}))\n    elif sys.argv[1:] == ["--check"]:',
    )
    (NEW / "controller.py").write_text(value)
    replace(NEW / "copy_cache.py", "890_000_000_000", "60_000_000_000")
    replace(NEW / "copy_cache.py", "850_000_000_000", "50_000_000_000")
    replace(
        NEW / "evaluate.py",
        'ROOT / "cached-image-evidence.json"',
        'ROOT / "image-import/result.json"',
    )
    replace(NEW / "evaluate.py", '== "node040"', '== "node090"')
    replace(NEW / "evaluate.py", "import signal\n", "import signal\nimport shutil\n")
    replace(
        NEW / "evaluate.py",
        "def generate(row, length, path, stream=True, logprob=False):\n",
        'def generate(row, length, path, stream=True, logprob=False):\n    assert shutil.disk_usage(ROOT).free > 50_000_000_000, "GPFS below temporary 50GB write reserve"\n',
    )
    replace(
        NEW / "import_worker_image.py",
        'out = Path(__file__).resolve().parent / os.environ["WORKER_ID"] / "image-import"',
        'out = Path(__file__).resolve().parent / "image-import"',
    )
    replace(
        NEW / "import_worker_image.py", '] in ("node205", "node207")', '] == "node090"'
    )
    job = yaml.safe_load((OLD / "eval-job.yaml").read_text())
    baseline = yaml.safe_load((SERVING / "replica-a-job.yaml").read_text())
    alias = "denovo-dspark-test-eval-20261104-r5"
    job["metadata"]["name"] = alias
    job["metadata"]["annotations"]["JobAlias"] = alias
    pod = job["spec"]["tasks"][0]["template"]["spec"]
    pod["nodeSelector"] = {
        "kubernetes.io/hostname": "node090",
        "node_shcheduleable": "true",
    }
    pod["containers"][0]["command"][-1] = REMOTE + "/run_eval.sh"
    init = baseline["spec"]["tasks"][0]["template"]["spec"]
    pod["volumes"] += [
        v
        for v in init["volumes"]
        if v["name"]
        in ("runtime", "host-ctr", "host-crictl", "host-snapshotter", "archive", "task")
    ]
    for v in pod["volumes"]:
        if v["name"] == "task":
            v["hostPath"]["path"] = REMOTE
    pod["initContainers"] = init["initContainers"]
    imported = pod["initContainers"][0]
    imported["command"][-1] = (
        "set -euo pipefail; umask 077; python3 -u "
        + REMOTE
        + "/import_worker_image.py > "
        + REMOTE
        + "/image-import/worker.log 2>&1"
    )
    imported["env"] = [e for e in imported["env"] if e["name"] != "WORKER_ID"]
    for e in imported["env"]:
        if e["name"] == "EXPECTED_NODE":
            e["value"] = "node090"
    for m in imported["volumeMounts"]:
        if m["name"] == "task":
            m["mountPath"] = REMOTE
    (NEW / "eval-job.yaml").write_text(yaml.safe_dump(job, sort_keys=False))
    protocol = json.loads((OLD / "protocol.json").read_text())
    protocol.update(
        {
            "headroom_bytes": 50_000_000_000,
            "resource_policy": "One8H100 node090 worker; verified init-only offline image import, regular container GPFS/shm only. Scheduling migration before any test; no numerical configuration changes.",
            "cached_nodes": [],
            "migration_from": "dspark-test-20261104-r4",
        }
    )
    protocol["resource_amendment"] = {
        "authorization": "先弄1组8卡跑起来; 允许本次放宽，优先跑起来",
        "prior_test_requests": 0,
        "temporary_gpfs_reserve_bytes": 50_000_000_000,
        "minimum_submit_free_bytes": 60_000_000_000,
    }
    protocol["cache_policy"] = protocol["cache_policy"].replace("r4", "r5")
    (NEW / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n"
    )
    (NEW / "resource-plan.json").write_text(
        json.dumps(
            {
                "node": "node090",
                "gpus": 8,
                "project": "WeaverRSI",
                "queue": "proj-32",
                "init_verifies_actual_eight_H100": True,
                "source_attempt": "r4",
                "sampling_and_inputs_unchanged": True,
            },
            indent=2,
        )
        + "\n"
    )
    (NEW / "RUN.md").write_text(
        "# Test r5: one eight-H100 scheduling migration\n\nUser prioritized starting one8GPU group and explicitly relaxed the850GB GPFS reserve for this run. Use node090 with init-only fixed image import and actual8H100 check. Worker generation, four variants, frozen inputs and ten EOS A/A diagnostics are unchanged from r4. Source r4 must be cancelled exactly while still Pending with no worker output, and its controller identity stopped before this attempt starts. No training restart. Temporary50GB GPFS write reserve; submit/cache-copy need60GB. Current serving replicas are preserved.\n"
    )
    names = json.loads((NEW / "archive-files.json").read_text())
    names += [
        "ctr_contract.py",
        "import_worker_image.py",
        "execution_contract.py",
        "resource-plan.json",
        "migrate_pending.py",
    ]
    (NEW / "archive-files.json").write_text(json.dumps(names, indent=2) + "\n")
    print(NEW)


if __name__ == "__main__":
    main()
