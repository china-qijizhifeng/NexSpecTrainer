#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export TRITON_CACHE_DIR="$RUN_DIR/cache/triton"
mkdir -p "$TRITON_CACHE_DIR"
set +e
python3 ops/yicloud-full/test_mhc_index64.py --baseline --output "$RUN_DIR/baseline.json" > "$RUN_DIR/baseline.log" 2>&1
baseline=$?
set -e
echo "$baseline" > "$RUN_DIR/baseline-exit.txt"
python3 ops/yicloud-full/test_mhc_index64.py --output "$RUN_DIR/RESULT.json" > "$RUN_DIR/patched.log" 2>&1
