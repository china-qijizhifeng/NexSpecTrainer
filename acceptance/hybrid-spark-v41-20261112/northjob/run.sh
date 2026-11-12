#!/bin/bash
set -u
TASK=/gpfs/users/linlizhi/hybrid-spark-kernel-20261112-northjob-r1
exec >"$TASK/worker.log" 2>&1
finish() {
    local rc=$?
    python3 - "$TASK" "$rc" <<'PYEND'
import datetime,json,pathlib,sys
out=pathlib.Path(sys.argv[1])
(out/'completion.json').write_text(json.dumps({'exit_code':int(sys.argv[2]),'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
PYEND
    exit "$rc"
}
trap finish EXIT
set -e
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export TRITON_CACHE_DIR="$TASK/triton-cache"
export CUDA_CACHE_PATH="$TASK/cuda-cache"
export PYTHONPATH="$TASK/source"
mkdir -p "$TASK/source" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
tar -xf "$TASK/source-3c6626d.tar" -C "$TASK/source"
cd "$TASK/source"
python3 - "$TASK" <<'PYRUNTIME'
import datetime,hashlib,json,pathlib,platform,subprocess,sys
import torch,triton,google.protobuf
from google.protobuf.internal.api_implementation import Type
out=pathlib.Path(sys.argv[1])
manifest=json.loads((out/'source-manifest.json').read_text())
for name,digest in manifest['files'].items():
    assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==digest, name
runtime={'source_sha':manifest['source_sha'],'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'triton':triton.__version__,'cuda':torch.version.cuda,'protobuf':google.protobuf.__version__,'protobuf_backend':Type(),'gpu_available':torch.cuda.is_available(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
if runtime['gpu_available']:
    runtime.update(gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()))
probe=subprocess.run(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True)
runtime.update(nvidia_smi_exit=probe.returncode,nvidia_smi_stdout=probe.stdout,nvidia_smi_stderr=probe.stderr)
(out/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
print(json.dumps(runtime),flush=True)
assert runtime['gpu_available'], 'No CUDA GPU allocated'
PYRUNTIME
python3 tests/gpu/hybrid_spark_acceptance.py --output "$TASK/kernel-results.json"
