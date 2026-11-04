"""Stop only the verified r6 controller and cancel its interrupted evaluation job."""

import json
import os
import time

from common import ROOT, write
from controller import api, single
from migrate_pending import open_verified_process, stop_verified_process

ALIAS = "denovo-dspark-test-eval-20261104-r6"
JOB = "job-b086ea04cfd9-20261104164429"


def main():
    observation = json.loads(
        (ROOT / "runtime-interruption-observation.json").read_text()
    )
    (observed,) = observation["exact_job"]
    assert observed["name"] == JOB and observed["status"] == "Pending"
    assert observation["eval-exit.txt"] == "0\n" and observation["result.json"] is None
    assert not (ROOT / "result.json").exists()
    assert not (ROOT / "target-only/aa-acceptance.json").exists()
    assert not any(
        ROOT.glob("*/test-*.json")
    ), "unexpected test output; inspect before cancelling"
    identity = single(ALIAS)
    assert identity and identity["name"] == JOB
    assert identity["status"] in ("Pending", "Running")
    assert identity["job_creator"] == "linlizhi" and identity["namespace"] == "qiji"
    record = json.loads((ROOT / "controller-process.json").read_text())
    assert record["pid"] == 8848 and str(record["start_ticks"]) == "2350852544"
    assert record["command"] == f"/usr/bin/python3 -u {ROOT}/controller.py "
    descriptor = open_verified_process(record)
    try:
        with (ROOT / "interruption-cancel-intent.json").open("x") as stream:
            json.dump(
                {
                    "time": time.time(),
                    "job": identity,
                    "controller": record,
                    "reason": "Running worker exited during A/A; platform returned job to Pending; no result or test output",
                    "single_use": True,
                },
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        stop_verified_process(descriptor)
    finally:
        os.close(descriptor)
    current = single(ALIAS)
    assert current and current["name"] == JOB
    write(
        ROOT / "interruption-before-delete.json",
        {
            "time": time.time(),
            "job": current,
            "execution_complete": False,
            "no_test_outputs": True,
        },
    )
    deleted = api("delete", alias=ALIAS)
    write(ROOT / "interruption-delete.json", deleted)
    assert single(ALIAS) is None
    receipt = {
        "time": time.time(),
        "job": identity,
        "controller_stopped": True,
        "alias_absent": True,
        "execution_complete": False,
        "test_requests_completed": 0,
        "gpu_aa_complete": False,
        "reason": "external runtime interruption; cause investigation retained separately",
        "delete": deleted,
        "automatic_retry": False,
    }
    write(ROOT / "interruption-cancellation.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
