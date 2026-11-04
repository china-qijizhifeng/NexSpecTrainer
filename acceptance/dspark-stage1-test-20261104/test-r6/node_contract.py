"""Allow scheduler-selected nodes while retaining resource and access boundaries."""

import re


def validate_node(node):
    assert isinstance(node, str) and re.fullmatch(
        r"[a-z0-9][a-z0-9.-]*", node
    ), "missing/invalid assigned node"


def validate_job(job):
    assert job["metadata"]["annotations"]["project"] == "WeaverRSI"
    assert job["spec"]["queue"] == "proj-32"
    assert job["spec"]["schedulerName"] == "volcano"
    (task,) = job["spec"]["tasks"]
    assert task["replicas"] == 1
    pod = task["template"]["spec"]
    assert pod["nodeSelector"] == {"node_shcheduleable": "true"}
    assert not pod.get("nodeName") and not pod.get(
        "affinity"
    ), "fixed node/affinity forbidden"
    assert pod["restartPolicy"] == "Never"
    assert pod["activeDeadlineSeconds"] == 86400
    assert sorted(pod["tolerations"], key=lambda row: row.get("value", "")) == [
        {
            "key": "task-type",
            "operator": "Equal",
            "value": "rsi-weaver",
            "effect": "NoSchedule",
        },
        {
            "key": "task-type",
            "operator": "Equal",
            "value": "weaver",
            "effect": "NoSchedule",
        },
    ]
    (worker,) = pod["containers"]
    expected = {"cpu": "64", "memory": "768Gi", "nvidia.com/gpu": "8"}
    assert worker["resources"]["requests"] == worker["resources"]["limits"] == expected
    # Northjob rewrites GPFS host mounts into its home/shared PVCs.
    assert all(
        "host" not in m["name"] and m["name"] != "runtime"
        for m in worker["volumeMounts"]
    )
    for container in [worker, *pod["initContainers"]]:
        assert not container.get("securityContext", {}).get("privileged", False)
        assert not any(e["name"] == "EXPECTED_NODE" for e in container.get("env", []))
        node_env = [e for e in container["env"] if e["name"] == "NODE_NAME"]
        assert node_env == [
            {
                "name": "NODE_NAME",
                "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
            }
        ]
