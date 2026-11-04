"""Batch CPU runner, protocol, placement, EOS and real process lifecycle checks."""

import json
import subprocess
import sys
from pathlib import Path

from common import sha, write


def main():
    root = Path(__file__).resolve().parent
    checks = []
    for name in (
        "check_performance.py",
        "check_dependencies.py",
        "check_identity.py",
        "check_observation.py",
        "check_scheduling.py",
        "check_lifecycle.py",
        "check_runner.py",
    ):
        result = subprocess.run(
            [sys.executable, str(root / name)],
            capture_output=True,
            text=True,
            check=True,
            timeout=180,
        )
        response = json.loads(result.stdout)
        assert response["passed"]
        checks.append({"script": name, "sha256": sha(root / name), **response})
    # Only the synthetic local HTTP fixture bypasses local disk headroom; actual
    # worker requests and controller retain GPFS50GB/submit60GB checks.
    driver = """import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import check_eos
check_eos.native_contracts(Path('contract-source'))
with patch('evaluate.shutil.disk_usage',return_value=SimpleNamespace(free=100_000_000_000)):
 check_eos.client_contracts(None)
print(json.dumps({'passed':True,'actual_pinned_eos_and_metrics':True,'real_http_sse_client':True,'disk_patch_fixture_only':True}))
"""
    result = subprocess.run(
        [sys.executable, "-c", driver],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    checks.append(
        {
            "script": "check_eos.py",
            "sha256": sha(root / "check_eos.py"),
            **json.loads(result.stdout),
        }
    )
    output = {
        "passed": True,
        "checks": checks,
        "python": sys.version,
        "scope": "CPU source and real HTTP/SSE/lifecycle checks; no GPU benchmark result",
    }
    write(root / "cpu-regression.json", output)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
