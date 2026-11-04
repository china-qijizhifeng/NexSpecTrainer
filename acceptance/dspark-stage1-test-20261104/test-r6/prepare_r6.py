"""Prepare an unbound eight-GPU attempt without modifying or cancelling r5."""

import json
import shutil
from pathlib import Path

import yaml

OLD = Path("/Users/shom/Downloads/dspark-test-20261104-r5")
NEW = OLD.with_name("dspark-test-20261104-r6")
REMOTE = "/gpfs/users/linlizhi/dspark-test-20261104-r6"
ARCHIVE = Path(__file__).resolve().parent


def replace(path, before, after):
    value = path.read_text()
    assert before in value, (path, before)
    path.write_text(value.replace(before, after))


def main():
    NEW.mkdir()
    names = json.loads((OLD / "archive-files.json").read_text())
    names.remove("migrate_pending.py")
    for name in names + [
        "inputs-manifest.json",
        "export-result.json",
        "r3-eos-reference.json",
        "cached-image-evidence.json",
        "execution-sha256.json",
        "monitor_supplemental.py",
    ]:
        shutil.copy2(OLD / name, NEW / name)
    for name in ("run_eval.sh", "monitor.py", "monitor_supplemental.py"):
        replace(NEW / name, "dspark-test-20261104-r5", "dspark-test-20261104-r6")
    replace(
        NEW / "controller.py",
        "denovo-dspark-test-eval-20261104-r5",
        "denovo-dspark-test-eval-20261104-r6",
    )
    replace(
        NEW / "controller.py",
        '    assert plan["node"] == "node090" and plan["gpus"] == 8',
        "    from node_contract import validate_job\n    import yaml\n\n"
        '    assert plan["node"] is None and plan["gpus"] == 8\n'
        '    assert plan["scheduling"] == "scheduler-selected"\n'
        '    validate_job(yaml.safe_load((ROOT / "eval-job.yaml").read_text()))',
    )
    replace(
        NEW / "controller.py",
        'previous = ROOT.parent / "dspark-test-20261104-r4"',
        'previous = ROOT.parent / "dspark-test-20261104-r5"',
    )
    replace(
        NEW / "controller.py",
        'single("denovo-dspark-test-eval-20261104-r4") is None',
        'single("denovo-dspark-test-eval-20261104-r5") is None',
    )
    replace(
        NEW / "controller.py",
        '"""Single-use EOS-aware evaluation on cached node040 image; reuse accepted r2 exports."""',
        '"""Single-use EOS-aware evaluation on a scheduler-selected node; reuse r2 exports."""',
    )
    replace(
        NEW / "import_worker_image.py",
        "from execution_contract import verify_execution",
        "from execution_contract import verify_execution\nfrom node_contract import validate_node",
    )
    replace(
        NEW / "import_worker_image.py",
        "    import os\n\n    assert (\n"
        '        os.environ["NODE_NAME"] == os.environ["EXPECTED_NODE"]\n'
        '        and os.environ["NODE_NAME"] == "node090"\n    )',
        '    validate_node(os.environ["NODE_NAME"])',
    )
    replace(
        NEW / "evaluate.py",
        '    assert (\n        imported["passed"] and imported["node"] == os.environ["NODE_NAME"] == "node090"\n    )',
        "    from node_contract import validate_node\n\n"
        '    validate_node(os.environ["NODE_NAME"])\n'
        '    assert imported["passed"] and imported["node"] == os.environ["NODE_NAME"]\n'
        '    assert len(imported["gpus"]) == 8 and all("H100" in row for row in imported["gpus"])',
    )
    replace(
        NEW / "evaluate.py",
        "        torch.cuda.get_device_capability(i) == (9, 0) for i in range(8)",
        "        torch.cuda.get_device_capability(i) == (9, 0)\n"
        '        and "H100" in torch.cuda.get_device_name(i) for i in range(8)',
    )
    job = yaml.safe_load((OLD / "eval-job.yaml").read_text())
    alias = "denovo-dspark-test-eval-20261104-r6"
    job["metadata"]["name"] = alias
    job["metadata"]["annotations"]["JobAlias"] = alias
    pod = job["spec"]["tasks"][0]["template"]["spec"]
    del pod["nodeSelector"]["kubernetes.io/hostname"]
    pod["containers"][0]["command"][-1] = REMOTE + "/run_eval.sh"
    for volume in pod["volumes"]:
        if volume["name"] == "task":
            volume["hostPath"]["path"] = REMOTE
    init = pod["initContainers"][0]
    init["command"][-1] = init["command"][-1].replace(str(OLD.name), str(NEW.name))
    init["env"] = [e for e in init["env"] if e["name"] != "EXPECTED_NODE"]
    for mount in init["volumeMounts"]:
        if mount["name"] == "task":
            mount["mountPath"] = REMOTE
    (NEW / "eval-job.yaml").write_text(yaml.safe_dump(job, sort_keys=False))
    protocol = json.loads((OLD / "protocol.json").read_text())
    protocol["cache_policy"] = protocol["cache_policy"].replace("r5", "r6")
    protocol.update(
        {
            "resource_policy": "One scheduler-selected worker; no hostname binding or hostname affinity. Same 8GPU/64CPU/768Gi. Init verifies actual8H100 and fixed image; regular container GPFS/shm only.",
            "migration_from": "dspark-test-20261104-r5",
            "scheduling_amendment": {
                "authorization": "为什么排队这么久？研究一下不要固定节点",
                "prior_test_requests": 0,
                "hostname_binding": False,
                "gpu_model_verified_in_init_and_worker": True,
            },
        }
    )
    (NEW / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n"
    )
    (NEW / "resource-plan.json").write_text(
        json.dumps(
            {
                "node": None,
                "scheduling": "scheduler-selected",
                "gpus": 8,
                "project": "WeaverRSI",
                "queue": "proj-32",
                "init_verifies_actual_eight_H100": True,
                "source_attempt": "r5",
                "sampling_and_inputs_unchanged": True,
            },
            indent=2,
        )
        + "\n"
    )
    (NEW / "RUN.md").write_text(
        "# Test r6: scheduler-selected eight H100s\n\n"
        "User requested removal of fixed-node scheduling. The r5 scheduler explicitly rejected node090 for insufficient memory against the unchanged768Gi request. Remove hostname selection, keep the allowed-node label and existing Weaver tolerations; do not bypass reserved partitions. Scheduler chooses the node; init and regular worker must independently verify actual8H100. Fixed OCI image import, source/model hashes, EOS protocol, ten A/A diagnostics and988 held-out requests are unchanged.\n\n"
        "Only migrate while exact r5 remains Pending0/1 with no init/worker output. Pin its verified controller with pidfd, stop it once, then recheck state, save cancellation receipts and cancel the exact job before launching r6 once. Never restart an old controller or modify an old execution manifest. Preserve serving replicas and all active data/checkpoint/cache dependencies. Temporary50GB reserve and60GB submit/cache threshold remain authorized.\n"
    )
    for name in ("node_contract.py", "migrate_pending.py", "check_scheduling.py"):
        shutil.copy2(ARCHIVE / name, NEW / name)
    names += [
        "migrate_pending.py",
        "node_contract.py",
        "check_scheduling.py",
        "monitor_supplemental.py",
    ]
    (NEW / "archive-files.json").write_text(json.dumps(names, indent=2) + "\n")
    print(NEW)


if __name__ == "__main__":
    main()
