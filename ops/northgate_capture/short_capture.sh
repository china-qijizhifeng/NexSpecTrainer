#!/bin/bash
set -euo pipefail
: "${RUN_DIR:?}" "${NORTHGATE_ROOT:?}"
test "${KUBEBRAIN_REPLICA:-0}" = 0
export NORTHGATE_CAPTURE_MODULE=ops.northgate_capture.short_capture
exec bash ops/northgate_capture/retest.sh
