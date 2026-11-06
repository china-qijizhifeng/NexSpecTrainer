#!/bin/bash
set -euo pipefail
: "${RUN_DIR:?}" "${NORTHGATE_ROOT:?}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
mkdir -p "$RUN_DIR/python-deps" "$RUN_DIR/cache"
export PYTHONPATH="$RUN_DIR/python-deps:$PWD:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_NO_SYNC=1
python3 -m pip install --no-index --no-deps --target "$RUN_DIR/python-deps" "$NORTHGATE_ROOT"/wheels/*.whl
cp -a plugins/sglang-spec-capture "$RUN_DIR/plugin-source"
python3 -m pip install --no-index --no-deps --no-build-isolation --target "$RUN_DIR/python-deps" "$RUN_DIR/plugin-source"
export SPECFORGE_SPEC_CAPTURE=1 SGLANG_PLUGINS=specforge_spec_capture NORTHGATE_LONG_CAPTURE=0
export SGLANG_SPEC_CAPTURE_GPU_PUT=0 SGLANG_SPEC_CAPTURE_WIRE_ENCODING=raw
export MOONCAKE_PROTOCOL=tcp MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551
export MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata
export MOONCAKE_GLOBAL_SEGMENT_SIZE=8589934592 MOONCAKE_LOCAL_BUFFER_SIZE=8589934592
export MC_STORE_MEMCPY=0 MC_TRANSFER_TIMEOUT=600
unset MC_FORCE_TCP MC_MS_AUTO_DISC MC_USE_TENT MC_USE_TEV1 MC_TE_FILTERS MOONCAKE_RDMA_DEVICES
export TRITON_CACHE_DIR="$RUN_DIR/cache/triton" CUDA_CACHE_PATH="$RUN_DIR/cache/cuda"
export XDG_CACHE_HOME="$RUN_DIR/cache/xdg" SGLANG_CACHE_DIR="$RUN_DIR/cache/sglang"
export DG_JIT_CACHE_DIR="$RUN_DIR/cache/deepgemm" TILELANG_CACHE_DIR="$RUN_DIR/cache/tilelang"
export FLASHINFER_WORKSPACE_BASE="$RUN_DIR/cache/flashinfer"
export PYTORCH_ALLOC_CONF=expandable_segments:True SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
# Clone the existing same-image cache; never modify old evidence.
capture_cache_source="${NORTHGATE_CACHE_RUN:-$NORTHGATE_ROOT/runs/northgate-v41-cp8-hybrid-32h200-20261106-r3}/worker-${KUBEBRAIN_REPLICA:-0}/cache"
if [ -z "${NORTHGATE_CACHE_RUN:-}" ]; then
    capture_cache_source="$NORTHGATE_ROOT/runs/northgate-v41-cp8-hybrid-32h200-20261106-r3/worker-0/cache"
fi
cp -a "$capture_cache_source/." "$RUN_DIR/cache/"
ulimit -l unlimited
python3 - <<'PY'
import json,os,torch,brotli,zstandard
from pathlib import Path
assert torch.cuda.device_count()==8
assert all('H200' in torch.cuda.get_device_name(i) for i in range(8))
Path(os.environ['RUN_DIR'],'environment.json').write_text(json.dumps({
 'torch':torch.__version__,'brotli':brotli.__version__,'zstandard':zstandard.__version__,
 'gpus':[torch.cuda.get_device_name(i) for i in range(8)],
 'env':{k:v for k,v in os.environ.items() if k.startswith(('SGLANG_','NORTHGATE_','MOONCAKE_','PYTORCH_'))}
},indent=2))
PY
exec python3 -m "${NORTHGATE_CAPTURE_MODULE:-ops.northgate_capture.retest}"
