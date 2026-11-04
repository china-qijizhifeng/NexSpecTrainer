"""Start only the new r7 controller after complete CPU and predecessor checks."""

import json
import subprocess
from pathlib import Path

ROOT = Path("/Users/shom/Downloads/dspark-test-20261104-r7")
PROGRAM = r"""
import json,runpy,sys
from pathlib import Path
r=Path('/gpfs/users/linlizhi/dspark-test-20261104-r7')
sys.path.insert(0,str(r))
checks=json.loads((r/'cpu-regression.json').read_text())
assert checks['passed'] and len(checks['checks'])==5
assert all(x['returncode']==0 for x in checks['checks'])
for name in ['controller-launch-intent.json','controller-started.json','eval-submit-intent.json','eval-job.json']:
    assert not (r/name).exists(), 'single-use launch already attempted; inspect, never rerun'
from common import verify_execution
verify_execution()
runpy.run_path(str(r/'launch_controller.py'),run_name='__main__')
"""


def main():
    # Claim this local entry as well as the remote launcher's existing open-x intent.
    with (ROOT / "activation-intent.json").open("x") as stream:
        json.dump({"single_use": True, "automatic_retry": False}, stream)
    response = subprocess.run(
        ["ssh", "linlizhi-bp-dev", "python3 -"],
        input=PROGRAM,
        capture_output=True,
        text=True,
        timeout=180,
    )
    (ROOT / "activation-response.json").write_text(
        json.dumps(
            {
                "returncode": response.returncode,
                "stdout": response.stdout,
                "stderr": response.stderr,
                "automatic_retry": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(response.stdout, end="")
    if response.returncode:
        print(response.stderr, end="")
        raise SystemExit(response.returncode)


if __name__ == "__main__":
    main()
