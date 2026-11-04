"""Read-only receipt monitor that preserves private init file permissions."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = r"""
import json,time,sys,urllib.parse
from pathlib import Path
root=Path("/gpfs/users/linlizhi/dspark-test-20261104-r6")
sys.path.insert(0,str(root))
from platform_ops import connection
out={"time":time.time(),"files":{}}
for name in ["image-import/gpu-preflight.json","image-import/result.json",
             "image-import/exit.txt","image-import/worker.log","image-import/import.log",
             "worker-cpu-regression.json","cache-copy-status.json","eval-environment.json",
             "target-only/aa-acceptance.json","target-only/r3-eos-repro.json"]:
    p=root/name
    if not p.exists():
        out["files"][name]=None
        continue
    info=p.stat()
    entry={"bytes":info.st_size,"mtime":info.st_mtime,"uid":info.st_uid}
    try:
        if p.suffix==".json":
            entry["value"]=json.loads(p.read_text())
        else:
            with p.open("rb") as stream:
                stream.seek(max(0,info.st_size-2000))
                entry["tail"]=stream.read().decode(errors="replace")
        entry["readable"]=True
    except PermissionError:
        entry.update(readable=False,reason="private init output; permissions preserved")
    except json.JSONDecodeError:
        entry.update(readable=True,reason="receipt being written; no mutation")
    out["files"][name]=entry
identity=json.loads((root/"eval-job.json").read_text())
params={"uuid":identity["name"],"kind":"vcjob","namespace":"qiji"}
out["overview"]=connection()("/api/task/get_task_overview?"+urllib.parse.urlencode(params))
print(json.dumps(out))
"""


def main():
    response = subprocess.run(
        ["ssh", "linlizhi-bp-dev", "python3 -"],
        input=SCRIPT,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    result = json.loads(response.stdout)
    (ROOT / "monitor-receipts-snapshot.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
