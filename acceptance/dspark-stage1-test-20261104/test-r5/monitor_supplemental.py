"""Read-only import and EOS gate receipts, separate from the frozen worker manifest."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = r"""
import json,time
from pathlib import Path
root=Path("/gpfs/users/linlizhi/dspark-test-20261104-r5")
names=["image-import/gpu-preflight.json","image-import/result.json",
       "worker-cpu-regression.json","cache-copy-status.json","eval-environment.json",
       "target-only/aa-acceptance.json","target-only/r3-eos-repro.json",
       "predecessor-cancellation.json"]
out={"time":time.time()}
for name in names:
    path=root/name
    out[name]=json.loads(path.read_text()) if path.exists() else None
for name in ["image-import/worker.log","image-import/import.log"]:
    path=root/name
    if path.exists():
        with path.open("rb") as stream:
            stream.seek(max(0,path.stat().st_size-2000))
            tail=stream.read().decode(errors="replace")
        out[name]={"bytes":path.stat().st_size,"mtime":path.stat().st_mtime,"tail":tail}
path=root/"image-import/exit.txt"
out["image-import/exit.txt"]=path.read_text().strip() if path.exists() else None
print(json.dumps(out))
"""


def main():
    result = subprocess.run(
        ["ssh", "linlizhi-bp-dev", "python3 -"],
        input=SCRIPT,
        text=True,
        capture_output=True,
        check=True,
        timeout=90,
    )
    data = json.loads(result.stdout)
    (ROOT / "monitor-supplemental-snapshot.json").write_text(
        json.dumps(data, indent=2) + "\n"
    )
    print(json.dumps(data))


if __name__ == "__main__":
    main()
