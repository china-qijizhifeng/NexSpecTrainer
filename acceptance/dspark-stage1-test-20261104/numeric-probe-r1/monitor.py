"""Read-only SSH monitor; writes snapshots only in the local evidence directory."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = "/gpfs/users/linlizhi/dspark-aa-numerics-20261104-r1"
script = r"""
import json,shutil,sys,time
from pathlib import Path
r=Path(REMOTE_ROOT)
sys.path.insert(0,str(r))
from controller import api,ALIASES
o={'time':time.time(),'free_bytes':shutil.disk_usage(r).free}
for name in ['cached-image-evidence.json','cache-copy-status.json','controller-process.json','pipeline-status.json','phase.json','export-job.json','export-terminal.json','export-delete.json','export-result.json',
             'eval-job.json','eval-terminal.json','eval-delete.json','eval-environment.json','environment.json','worker-started.json','eval-interrupted.json','eval-failure.json','result.json']:
 p=r/name
 o[name]=json.loads(p.read_text()) if p.exists() else None
o['jobs']={s:api('jobs',alias=a) for s,a in ALIASES.items()}
rec=o['controller-process.json']
if rec:
 p=Path(f"/proc/{rec['pid']}/stat")
 o['controller_verified']=p.exists() and p.read_text().rsplit(')',1)[1].split()[19]==str(rec['start_ticks']) and Path(f"/proc/{rec['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode()==rec['command']
for name in ['export-exit.txt','eval-exit.txt']:
 p=r/name
 o[name]=p.read_text().strip() if p.exists() else None
for name in ['controller.log','export-worker.log','eval-worker.log']+[g+'/server.log' for g in ('baseline','no-fused-allreduce')]:
 p=r/name
 if p.exists():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-3000));tail=f.read().decode(errors='replace')
  o[name]={'mtime':p.stat().st_mtime,'bytes':p.stat().st_size,'tail':tail}
o['summaries']={g:json.loads((r/g/'summary.json').read_text()) for g in ('baseline','no-fused-allreduce') if (r/g/'summary.json').exists()}
print(json.dumps(o))
"""
script = "REMOTE_ROOT=" + repr(REMOTE) + "\n" + script
result = subprocess.run(
    ["ssh", "linlizhi-bp-dev", "python3", "-"],
    input=script.encode(),
    capture_output=True,
    check=True,
)
o = json.loads(result.stdout)
(ROOT / "monitor-snapshot.json").write_text(json.dumps(o, indent=2) + "\n")
for name in [
    "controller-process.json",
    "pipeline-status.json",
    "phase.json",
    "export-job.json",
    "eval-job.json",
    "export-result.json",
    "result.json",
    "export-terminal.json",
    "eval-terminal.json",
    "export-delete.json",
    "eval-delete.json",
]:
    if o.get(name) is not None:
        (ROOT / name).write_text(json.dumps(o[name], indent=2) + "\n")
print(
    json.dumps(
        {
            k: o.get(k)
            for k in [
                "time",
                "free_bytes",
                "controller_verified",
                "jobs",
                "pipeline-status.json",
                "phase.json",
                "export-exit.txt",
                "eval-exit.txt",
                "eval-failure.json",
            ]
        }
    )
)
