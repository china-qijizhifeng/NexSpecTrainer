"""One explicit job identity must agree across YAML, controller and worker root."""

from common import ROOT

EXPECTED_ALIAS = "denovo-dspark-test-eval-20261104-r9"
REMOTE_ROOT = "/gpfs/users/linlizhi/dspark-test-20261104-r9"


def validate_identity(job, aliases):
    assert ROOT.name == "dspark-test-20261104-r9"
    assert aliases == {"eval": EXPECTED_ALIAS}
    assert job["metadata"]["name"] == EXPECTED_ALIAS
    assert job["metadata"]["annotations"]["JobAlias"] == EXPECTED_ALIAS
    assert job["metadata"]["namespace"] == "qiji"
    (task,) = job["spec"]["tasks"]
    pod = task["template"]["spec"]
    (worker,) = pod["containers"]
    assert worker["command"] == ["/bin/bash", REMOTE_ROOT + "/run_eval.sh"]
    task_volume = next(v for v in pod["volumes"] if v["name"] == "task")
    assert task_volume["hostPath"]["path"] == REMOTE_ROOT
    (init,) = pod["initContainers"]
    assert REMOTE_ROOT + "/import_worker_image.py" in init["command"][-1]
    assert REMOTE_ROOT + "/image-import/invocation-started" in init["command"][-1]
    assert any(
        v["name"] == "task" and v["mountPath"] == REMOTE_ROOT
        for v in init["volumeMounts"]
    )
    forbidden = ("dspark-test-20261104-r8", "denovo-dspark-test-eval-20261104-r7")
    assert not any(name in str(job) for name in forbidden)


def validate_api_row(row):
    assert row["user_def_name"] == EXPECTED_ALIAS
    assert row["namespace"] == "qiji" and row["job_creator"] == "linlizhi"
    assert row["user_gpu"] == 8
