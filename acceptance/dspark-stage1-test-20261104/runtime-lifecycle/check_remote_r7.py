"""Run pre-submit CPU checks using serving fixtures verified against frozen source pins."""

import hashlib
import io
import json
import shlex
import subprocess
import tarfile
from pathlib import Path

ROOT = Path("/Users/shom/Downloads/dspark-test-20261104-r7")
REMOTE = "/gpfs/users/linlizhi/dspark-test-20261104-r7"


def main():
    pins = json.loads((ROOT / "serving-source-pins.json").read_text())
    content = io.BytesIO()
    with tarfile.open(fileobj=content, mode="w:gz") as archive:
        for name in (
            "srt/managers/schedule_batch.py",
            "srt/speculative/spec_decoding_metrics.py",
        ):
            path = ROOT / "contract-source" / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == pins[name]
            archive.add(path, arcname="contract-source/" + name)
    program = f"""
import hashlib,json,subprocess,sys,tarfile,time,shutil
from pathlib import Path
r=Path({REMOTE!r})
previous=r/'cpu-regression.json'
assert previous.exists()
previous.rename(r/'cpu-precheck-invalid-path.json')
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as archive:
    for member in archive:
        assert member.isfile() and member.name.startswith('contract-source/')
        assert '..' not in Path(member.name).parts
        target=r/member.name
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:
            stream.write(archive.extractfile(member).read())
pins=json.loads((r/'serving-source-pins.json').read_text())
fixtures={{}}
for p in (r/'contract-source').rglob('*.py'):
    name=str(p.relative_to(r/'contract-source'))
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest==pins[name]
    fixtures[name]=digest
checks=[]
commands=[[sys.executable,'check_runner.py'],
          [sys.executable,'check_eos.py','--serving-source',str(r/'contract-source')],
          [sys.executable,'check_scheduling.py'],[sys.executable,'check_lifecycle.py'],
          [sys.executable,'controller.py','--prepare-check']]
for command in commands:
    p=subprocess.run(command,cwd=r,capture_output=True,text=True)
    entry={{'command':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}}
    checks.append(entry)
    receipt={{'passed':all(x['returncode']==0 for x in checks),'checks':checks,
             'time':time.time(),'free_bytes':shutil.disk_usage(r).free,
             'serving_fixture_sha256':fixtures,'scope':'actual CPU EOS/SSE/lifecycle and API preflight; no GPU result',
             'harness_correction':'source overlay is NexSpecTrainer, not SGLang; use exact two pinned serving files. Prior invalid-path receipt retained. Production50GB reserve unchanged.'}}
    (r/'cpu-regression.json').write_text(json.dumps(receipt,indent=2)+chr(10))
    print(json.dumps(entry),flush=True)
    assert p.returncode==0,'pre-submit regression failed; do not launch'
print(json.dumps({{'passed':True,'cpu_checks':len(checks),'submitted':False}}))
"""
    result = subprocess.run(
        ["ssh", "linlizhi-bp-dev", "python3 -c " + shlex.quote(program)],
        input=content.getvalue(),
        capture_output=True,
        timeout=180,
    )
    print(result.stdout.decode(), end="")
    if result.returncode:
        print(result.stderr.decode(), end="")
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
