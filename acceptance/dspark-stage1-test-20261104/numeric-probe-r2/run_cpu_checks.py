"""Run the full CPU contract batch and record checked script bytes before freezing."""

import json
import subprocess
import sys
from pathlib import Path

from common import sha, write


def main():
    root = Path(__file__).resolve().parent
    checks = []
    for name, extra in (
        ("check_permissions.py", []),
        ("check_probe.py", ["--serving-source", str(root / "contract-source")]),
        ("check_lifecycle.py", []),
        ("check_runner.py", []),
        ("check_eos.py", ["--serving-source", str(root / "contract-source")]),
    ):
        response = subprocess.run(
            [sys.executable, str(root / name), *extra],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        result = json.loads(response.stdout)
        assert result["passed"]
        checks.append({"script": name, "sha256": sha(root / name), **result})
    if root.as_posix().startswith("/gpfs/"):
        assert next(item for item in checks if item["script"] == "check_probe.py")[
            "actual_private_validation_plan_checked"
        ]
    result = {
        "passed": True,
        "checks": checks,
        "python": sys.version,
        "scope": "CPU source/HTTP/SSE/logprob/resource/lifecycle checks; no GPU inference",
    }
    write(root / "cpu-regression.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
