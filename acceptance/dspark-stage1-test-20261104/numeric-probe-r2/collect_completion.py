"""Read-only completion audit; collect only allowlisted public control receipts."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = "/gpfs/users/linlizhi/dspark-aa-numerics-20261104-r2"
SCRIPT = r"""
import hashlib,json,os,stat,sys,time
from pathlib import Path
r=Path(REMOTE)
sys.path.insert(0,str(r))
from controller import api,ALIASES
names=['result.json','phase.json','pipeline-status.json','eval-terminal.json',
       'eval-delete.json','controller-process.json','worker-started.json','environment.json']
names += [mode+'/'+name for mode in ('baseline','no-fused-allreduce')
          for name in ('resolved.json','summary.json','launch.json','server-process.json')]
files={name:json.loads((r/name).read_text()) for name in names}
assert files['result.json']['probe_complete']
assert files['result.json']['request_audit']=={'passed':True,'requests':24}
assert files['result.json']['held_out_requests']==0
assert files['result.json']['numerical_fix_validated'] is False
terminal=files['eval-terminal.json'];deleted=files['eval-delete.json']
assert terminal['name']==deleted['name']=='job-5c838f4eb38e-20261104200236'
assert terminal['user_def_name']==deleted['alias']==ALIASES['eval']
assert terminal['status']=='Completed' and deleted['response']['success']
assert (r/'eval-exit.txt').read_text().strip()=='0'
assert not (r/'eval-failure.json').exists()
assert not (r/'eval-interrupted.json').exists()
jobs={stage:api('jobs',alias=alias) for stage,alias in ALIASES.items()}
assert all(not found for found in jobs.values())
for mode in ('baseline','no-fused-allreduce'):
 assert files[mode+'/summary.json']==files['result.json']['modes'][mode]
a=files['baseline/resolved.json'];b=files['no-fused-allreduce/resolved.json']
assert a['enforce_disable_flashinfer_allreduce_fusion'] is False
assert a['flashinfer_allreduce_fusion_backend']=='auto'
assert b['enforce_disable_flashinfer_allreduce_fusion'] is True
assert b['flashinfer_allreduce_fusion_backend'] is None
assert {k for k in a if a[k]!=b[k]}=={
 'enforce_disable_flashinfer_allreduce_fusion','flashinfer_allreduce_fusion_backend'}
old=files['baseline/launch.json']['command'];new=files['no-fused-allreduce/launch.json']['command']
assert new==old+['--enforce-disable-flashinfer-allreduce-fusion']
manifest_audits={}
for root,count,expected in ((r,33,'e0c85a2eed5adf12002067a88a2dad47e95d4ad86e396aee55c8e94d40569d79'),
 (Path('/gpfs/users/linlizhi/dspark-test-20261104-r7'),860,
 '38a080efa6a37603725d8e0f0cbce67dc884ee8cd97b03ecf0950922880ed4e2')):
 manifest=root/'execution-sha256.json'
 assert hashlib.sha256(manifest.read_bytes()).hexdigest()==expected
 rows=json.loads(manifest.read_text());assert len(rows)==count
 for name,digest in rows.items():
  assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
 manifest_audits[root.name]={'entries':count,'sha256':expected,'all_files_verified':True}
private=[]
for mode in ('baseline','no-fused-allreduce'):
 for path in sorted((r/mode).glob('request-*')):
  s=path.stat()
  assert stat.S_IMODE(s.st_mode)==0o600
  private.append({'path':str(path.relative_to(r)),'bytes':s.st_size,
                  'mode':oct(stat.S_IMODE(s.st_mode)),'uid':s.st_uid,
                  'readable_by_ssh_uid':os.access(path,os.R_OK)})
assert len(private)==48
rec=files['controller-process.json'];p=Path('/proc')/str(rec['pid'])
verified=(p/'stat').exists() and (p/'stat').read_text().rsplit(')',1)[1].split()[19]==str(rec['start_ticks']) and (p/'cmdline').read_bytes().replace(b'\0',b' ').decode()==rec['command']
audit={'passed':True,'time':time.time(),'terminal':terminal,'delete':deleted,
       'jobs':jobs,'all_aliases_absent':True,'controller_verified':verified,
       'exit_code':0,'request_audit':files['result.json']['request_audit'],
       'manifest_audits':manifest_audits,'effective_config_intervention_verified':True,
       'public_receipts_mode':{name:oct(stat.S_IMODE((r/name).stat().st_mode)) for name in names},
       'private_capture_stat':private,'posthoc_private_replay_performed':False,
       'held_out_requests':0,'numerical_root_cause_confirmed':False,
       'numerical_fix_validated':False}
print(json.dumps({'files':files,'audit':audit},allow_nan=False))
"""


def main():
    script = "REMOTE=" + repr(REMOTE) + "\n" + SCRIPT
    reply = subprocess.run(
        ["ssh", "linlizhi-bp-dev", "python3", "-"],
        input=script.encode(),
        capture_output=True,
        check=True,
    )
    report = json.loads(reply.stdout)
    for name, value in report["files"].items():
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    (ROOT / "completion-audit.json").write_text(
        json.dumps(report["audit"], indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: report["audit"][k]
                for k in (
                    "passed",
                    "all_aliases_absent",
                    "controller_verified",
                    "request_audit",
                    "effective_config_intervention_verified",
                    "held_out_requests",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
