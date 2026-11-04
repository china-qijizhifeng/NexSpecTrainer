"""Install frozen r7 files once; retain source, inputs and exports as read-only dependencies."""

import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path("/Users/shom/Downloads/dspark-test-20261104-r7")
REMOTE = "/gpfs/users/linlizhi/dspark-test-20261104-r7"


def main():
    names = json.loads((ROOT / "archive-files.json").read_text()) + [
        "archive-files.json",
        "execution-sha256.json",
        "source-provenance.json",
    ]
    content = io.BytesIO()
    with tarfile.open(fileobj=content, mode="w:gz") as archive:
        for name in names:
            archive.add(ROOT / name, arcname=name)
    program = f"""
import json,sys,tarfile
from pathlib import Path
root=Path({REMOTE!r})
root.mkdir()
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as archive:
    for member in archive:
        assert member.isfile() and '/' not in member.name
        with (root/member.name).open('xb') as target:
            target.write(archive.extractfile(member).read())
for name in ['source','export-step25','export-step118','export-step25-verification.json',
             'export-step118-verification.json','test-inputs.jsonl','validation-inputs.jsonl']:
    dependency=root.parent/'dspark-test-20261104-r2'/name
    assert dependency.exists(),name
    (root/name).symlink_to(dependency)
(root/'image-import').mkdir()
sys.path.insert(0,str(root))
from common import verify_execution
verify_execution()
print(json.dumps({{'deployed':True,'execution_verified':True,'submitted':False}}))
"""
    # The shell receives only fixed public program text. No credentials or tokens.
    import shlex

    result = subprocess.run(
        ["ssh", "linlizhi-bp-dev", "python3 -c " + shlex.quote(program)],
        input=content.getvalue(),
        capture_output=True,
        check=True,
        timeout=120,
    )
    print(result.stdout.decode(), end="")


if __name__ == "__main__":
    main()
