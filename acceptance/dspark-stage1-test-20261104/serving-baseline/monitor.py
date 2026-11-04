"""Observe exact owned baseline jobs and redacted GPFS receipts; never mutate remote state."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = r"""
import json, shutil, sys, time
from pathlib import Path
sys.path.insert(0, '/gpfs/users/linlizhi/dspark-test-20261104-r4')
from controller import api
root = Path('/gpfs/users/linlizhi/dspark-v41-northgate-20261104-r1')
out = {'time': time.time(), 'free_bytes': shutil.disk_usage(root).free, 'workers': {}}
for worker, alias in [('replica-a', 'dspark-v41-baseline-a-20261104-r1'), ('replica-b', 'dspark-v41-baseline-b-20261104-r1')]:
    row = {'alias': alias, 'jobs': api('jobs', alias=alias)}
    for name in ['job.json', 'phase.json', 'environment.json', 'smoke.json', 'failure.json', 'server-exit.json', 'image-import/gpu-preflight.json', 'image-import/result.json', 'terminal.json', 'delete.json', 'api-acceptance.json', 'api-acceptance-failure.json', 'resolved-config-acceptance.json', 'service-identity.json', 'service-detail.json', 'submission-preflight.json']:
        path = root / worker / name
        if path.exists():
            row[name] = json.loads(path.read_text())
    for name in ['worker.log', 'server.log', 'image-import/worker.log', 'image-import/import.log', 'image-import/exit.txt']:
        path = root / worker / name
        if path.exists():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 5000))
                tail = stream.read().decode(errors='replace')
            row[name] = {'bytes': path.stat().st_size, 'mtime': path.stat().st_mtime, 'tail': tail}
    out['workers'][worker] = row
for name in ['northgate-publication.json', 'final-acceptance.json', 'github-main-publication.json']:
    path = root / name
    if path.exists():
        out[name] = json.loads(path.read_text())
print(json.dumps(out))
"""


def main():
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "linlizhi-bp-dev", "python3", "-"],
        input=SOURCE,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    result = json.loads(completed.stdout)
    (ROOT / "monitor-snapshot.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, worker in result["workers"].items():
        for field, value in worker.items():
            if field.endswith(".json"):
                path = ROOT / name / field
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                "time": result["time"],
                "free_bytes": result["free_bytes"],
                "workers": {
                    name: {
                        key: row.get(key)
                        for key in [
                            "jobs",
                            "phase.json",
                            "image-import/gpu-preflight.json",
                            "image-import/result.json",
                            "failure.json",
                        ]
                    }
                    for name, row in result["workers"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
