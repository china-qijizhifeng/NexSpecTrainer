#!/bin/bash
set -euo pipefail
: "${RUN_DIR:?}" "${NORTHGATE_ROOT:?}" "${CODEC_BENCH_INPUT:?}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
mkdir -p "$RUN_DIR/python-deps"
python3 -m pip install --no-index --no-deps --target "$RUN_DIR/python-deps" "$NORTHGATE_ROOT"/wheels/*.whl
export PYTHONPATH="$RUN_DIR/python-deps:$PWD:${PYTHONPATH:-}"
extra=()
if [ "${CODEC_BENCH_NATIVE:-0}" = 1 ]; then
    extra+=(--native-comparison)
fi
exec python3 -m ops.northgate_capture.codec_bench --input "$CODEC_BENCH_INPUT" --output "$RUN_DIR/benchmark" "${extra[@]}"
