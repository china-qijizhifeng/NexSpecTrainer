#!/bin/bash
set -euo pipefail
TASK_ROOT=/gpfs/users/linlizhi/dspark-test-20261104-r4
exec >"$TASK_ROOT/eval-worker.log" 2>&1
trap 'rc=$?; echo "$rc" > "$TASK_ROOT/eval-exit.txt"' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_RAGGED_VERIFY_MODE=static
unset SGLANG_PLUGINS SPECFORGE_SPEC_CAPTURE SGLANG_SIMULATE_ACC_LEN
ulimit -l unlimited
python3 "$TASK_ROOT/copy_cache.py"
python3 "$TASK_ROOT/with_kernel_cache.py" --cache-root "$TASK_ROOT/cache" --namespace sglang-c6da252c6837-sm90 -- python3 "$TASK_ROOT/evaluate.py"
