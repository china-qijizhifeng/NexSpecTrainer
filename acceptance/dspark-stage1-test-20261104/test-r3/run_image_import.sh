#!/bin/bash
set -euo pipefail
TASK_ROOT=/gpfs/users/linlizhi/dspark-test-20261104-r3
mkdir -p "$TASK_ROOT/image-import"
exec >"$TASK_ROOT/image-import/worker.log" 2>&1
trap 'rc=$?; echo "$rc" > "$TASK_ROOT/image-import/exit.txt"' EXIT
python3 -u "$TASK_ROOT/import_worker_image.py"
