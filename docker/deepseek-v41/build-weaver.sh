#!/bin/bash
set -euo pipefail
MODE=${1:?specify prepare, DeepGEMM or FlashMLA}
ROOT=/opt/nexspec
if [ "$MODE" = prepare ]; then
    mkdir -p "$ROOT/src" "$ROOT/wheels" "$ROOT/evidence"
    python - <<'PY'
from pathlib import Path
import hashlib, json, subprocess
root = Path('/build')
manifest = json.loads((root/'native-inputs.json').read_text())
for filename, digest in manifest['sha256'].items():
    with (root/filename).open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == digest, filename
subprocess.run(['tar', '-xf', '/build/native-sources.tar.gz', '-C', '/opt/nexspec/src'], check=True)
subprocess.run(['tar', '-xf', '/build/elfutils-dev-headers.tar.gz', '-C', '/'], check=True)
setup = Path('/opt/nexspec/src/FlashMLA/setup.py')
text = setup.read_text()
start = text.index("try:\n    cmd = ['git', 'rev-parse', '--short', 'HEAD']")
end = text.index('\n\nsetup(', start)
setup.write_text(text[:start] + 'rev = ' + repr('+' + manifest['revisions']['FlashMLA'][:12]) + text[end:])
Path('/opt/nexspec/evidence/native-inputs.json').write_text(json.dumps(manifest, indent=2))
PY
    exit 0
fi
case "$MODE" in DeepGEMM|FlashMLA) ;; *) exit 2 ;; esac
export CPATH="$CUDA_HOME/include:$CUDA_HOME/include/cccl${CPATH:+:$CPATH}"
# FlashMLA's host C++ unit includes CUTLASS and needs CCCL explicitly;
# nvcc discovers it implicitly, while the host compiler does not.
printf '#include <cuda_runtime.h>\n#include <cuda/std/utility>\n' \
    > "$ROOT/evidence/weaver-cuda-headers.cpp"
g++ -std=c++20 -E "$ROOT/evidence/weaver-cuda-headers.cpp" \
    > "$ROOT/evidence/weaver-cuda-headers.i"
python - <<'PY'
import torch, importlib.metadata as m, json
from pathlib import Path
import transformer_engine.pytorch
assert torch.__version__ == '2.10.0a0+b558c986e8.nv25.11', torch.__version__
assert m.version('transformer-engine') == '2.20.2+6ea2a74'
result = {'torch': torch.__version__, 'cuda': torch.version.cuda,
          'transformer_engine': m.version('transformer-engine'),
          'torch_cxx11abi': torch._C._GLIBCXX_USE_CXX11_ABI}
Path('/opt/nexspec/evidence/weaver-build-abi.json').write_text(json.dumps(result, indent=2))
print(result)
PY
nvcc --version
python -m pip wheel --no-build-isolation --no-deps --wheel-dir "$ROOT/wheels" "$ROOT/src/$MODE"
python - <<'PY'
from pathlib import Path
import hashlib, json
rows = []
for p in sorted(Path('/opt/nexspec/wheels').glob('*.whl')):
    with p.open('rb') as f:
        rows.append({'filename': p.name, 'sha256': hashlib.file_digest(f, 'sha256').hexdigest(), 'size': p.stat().st_size})
Path('/opt/nexspec/evidence/native-wheels.json').write_text(json.dumps(rows, indent=2))
PY
