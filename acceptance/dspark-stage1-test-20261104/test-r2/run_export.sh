#!/bin/bash
set -euo pipefail
TASK_ROOT=/gpfs/users/linlizhi/dspark-test-20261104-r2
exec >"$TASK_ROOT/export-worker.log" 2>&1
trap 'rc=$?; echo "$rc" > "$TASK_ROOT/export-exit.txt"' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NATIVE_COMPAT=/gpfs/users/linlizhi/nexspec-hopper-validation-20261102/compat-weaver-r2
python3 -m pip install --no-index --no-deps --target /tmp/dspark-test-deps /gpfs/users/linlizhi/nexspec-v41-rdma-realtrace-20261103/wheels/*.whl
export PYTHONPATH="$NATIVE_COMPAT/overlay/nvidia_cutlass_dsl/dsl_packages:$NATIVE_COMPAT/overlay:/gpfs/users/linlizhi/dspark-denovo-specforge-stage1-20261104-r6/train-core:$NATIVE_COMPAT/tile:$TASK_ROOT/source:/gpfs/users/linlizhi/nfc-sharded-pipeline-20261103/source-cpuview-r2/src:/tmp/dspark-test-deps"
python3 "$TASK_ROOT/export_models.py"
