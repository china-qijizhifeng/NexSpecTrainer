"""Check placement, access boundaries and the512Gi scheduler contract."""

import copy
import json
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
    print(
        json.dumps(
            {
                "passed": True,
                "scheduler_selected": True,
                "forbidden_placement_and_tolerations_rejected": True,
                "memory_512Gi": True,
            }
        )
    )


if __name__ == "__main__":
    main()
