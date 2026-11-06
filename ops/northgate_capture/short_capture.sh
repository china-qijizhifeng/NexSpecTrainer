#!/bin/bash
set -euo pipefail
: "${RUN_DIR:?}" "${NORTHGATE_ROOT:?}"
export NORTHGATE_CAPTURE_RUN_DIR="$RUN_DIR"
export RUN_DIR="$RUN_DIR/worker-${KUBEBRAIN_REPLICA:-0}"
export NORTHGATE_CAPTURE_MODULE=ops.northgate_capture.short_capture
exec bash ops/northgate_capture/retest.sh
