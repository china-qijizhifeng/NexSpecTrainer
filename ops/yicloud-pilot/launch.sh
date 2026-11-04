#!/bin/bash
set -euo pipefail
: "${RUN_DIR:?}" "${MIGRATION_ROOT:?}"
REPO_ROOT=$PWD
export REPO_ROOT PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_NO_SYNC=1
export PYTHONPATH="$REPO_ROOT:$MIGRATION_ROOT/runtime-deps/train-core:$MIGRATION_ROOT/runtime-deps/nfc-src:/opt/nexspec/overlay/nvidia_cutlass_dsl/dsl_packages:/opt/nexspec/overlay:/opt/nexspec/tile:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="$RUN_DIR/cache/triton" CUDA_CACHE_PATH="$RUN_DIR/cache/cuda" XDG_CACHE_HOME="$RUN_DIR/cache/xdg"
export DG_JIT_CACHE_DIR="$RUN_DIR/cache/deepgemm" TILELANG_CACHE_DIR="$RUN_DIR/cache/tilelang"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$DG_JIT_CACHE_DIR" "$TILELANG_CACHE_DIR"
export NCCL_DEBUG=WARN TORCH_FR_BUFFER_SIZE=20000 TORCH_NCCL_DUMP_ON_TIMEOUT=1
# Select the tested source bundles; preserve image Torch/TE/CUDA unchanged.
if ! python3 -c 'import zstandard' >/dev/null 2>&1; then
  python3 -m pip install --no-index --no-deps --target "$RUN_DIR/python-deps" "$MIGRATION_ROOT"/runtime-deps/wheels/*.whl
  export PYTHONPATH="$RUN_DIR/python-deps:$PYTHONPATH"
fi
python3 "$REPO_ROOT/ops/yicloud-pilot/prepare.py"
python3 "$REPO_ROOT/.agents/skills/training-run/scripts/preflight.py" --output "$RUN_DIR/environment.json"
if [[ ${PILOT_CPU_ONLY:-0} == 1 ]]; then
  python3 "$REPO_ROOT/ops/yicloud-pilot/test_config.py" -v
  exit 0
fi
python3 - <<'PY'
import torch,shutil,json,os
from pathlib import Path
assert torch.cuda.device_count()==8
assert all(torch.cuda.get_device_capability(i)==(9,0) for i in range(8))
assert shutil.disk_usage('/dev/shm').free>=288*1024**3,'NFC requires at least 288 GiB available /dev/shm'
from specforge.modeling.draft.deepseek_v41.validation import preflight
preflight()
Path(os.environ['RUN_DIR'],'gpu-environment.json').write_text(json.dumps({'gpus':[torch.cuda.get_device_name(i) for i in range(8)],'shm_free':shutil.disk_usage('/dev/shm').free}))
PY
export PILOT_PHASE=initial
python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 "$REPO_ROOT/ops/yicloud-pilot/audit_train.py" train --config "$RUN_DIR/initial.yaml"
python3 "$REPO_ROOT/ops/yicloud-pilot/verify.py" initial
export PILOT_PHASE=resume
python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 "$REPO_ROOT/ops/yicloud-pilot/audit_train.py" train --config "$RUN_DIR/config.yaml"
python3 "$REPO_ROOT/ops/yicloud-pilot/verify.py" final
