"""Reproduce the reused-alias mistake and reject every neighboring identity mismatch."""

import copy
import json
from pathlib import Path

import controller
import yaml
from identity_contract import EXPECTED_ALIAS, validate_api_row, validate_identity


def rejected(call):
    try:
        call()
    except AssertionError:
        return
    raise AssertionError("inconsistent job identity accepted")


def main():
    root = Path(__file__).resolve().parent
    job = yaml.safe_load((root / "eval-job.yaml").read_text())
    validate_identity(job, controller.ALIASES)
    old = yaml.safe_load(
        (root.parent / "dspark-test-20261104-r8/eval-job.yaml").read_text()
    )
    rejected(lambda: validate_identity(old, controller.ALIASES))
    for field in (
        "name",
        "annotation",
        "worker-command",
        "task-volume",
        "init-command",
        "init-mount",
    ):
        altered = copy.deepcopy(job)
        pod = altered["spec"]["tasks"][0]["template"]["spec"]
        if field == "name":
            altered["metadata"]["name"] = "denovo-dspark-test-eval-20261104-r7"
        elif field == "annotation":
            altered["metadata"]["annotations"][
                "JobAlias"
            ] = "denovo-dspark-test-eval-20261104-r8"
        elif field == "worker-command":
            pod["containers"][0]["command"][
                -1
            ] = "/gpfs/users/linlizhi/dspark-test-20261104-r8/run_eval.sh"
        elif field == "task-volume":
            next(v for v in pod["volumes"] if v["name"] == "task")["hostPath"][
                "path"
            ] = "/gpfs/users/linlizhi/dspark-test-20261104-r8"
        elif field == "init-command":
            pod["initContainers"][0]["command"][
                -1
            ] = "python3 /gpfs/users/linlizhi/dspark-test-20261104-r8/import_worker_image.py"
        elif field == "init-mount":
            next(
                v
                for v in pod["initContainers"][0]["volumeMounts"]
                if v["name"] == "task"
            )["mountPath"] = "/gpfs/users/linlizhi/dspark-test-20261104-r8"
        rejected(lambda: validate_identity(altered, controller.ALIASES))
    rejected(
        lambda: validate_identity(job, {"eval": "denovo-dspark-test-eval-20261104-r7"})
    )
    row = {
        "name": "fixture-job",
        "user_def_name": EXPECTED_ALIAS,
        "namespace": "qiji",
        "job_creator": "linlizhi",
        "user_gpu": 8,
    }
    validate_api_row(row)
    for key, value in (
        ("user_def_name", "denovo-dspark-test-eval-20261104-r7"),
        ("namespace", "default"),
        ("job_creator", "someone-else"),
        ("user_gpu", 1),
    ):
        altered = {**row, key: value}
        rejected(lambda: validate_api_row(altered))
    print(
        json.dumps(
            {
                "passed": True,
                "original_r8_wrong_alias_reproduced": True,
                "yaml_alias_controller_command_mount_api_identity_checked": True,
                "negative_mutations_rejected": 11,
                "gpu_attempt_submitted": False,
            }
        )
    )


if __name__ == "__main__":
    main()
