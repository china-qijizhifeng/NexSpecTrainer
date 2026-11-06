#!/bin/bash
set -euo pipefail
: "${RUN_DIR:?}" "${KUBEBRAIN_REPLICA:?}"
export NORTHGATE_CAPTURE_RUN_DIR="$RUN_DIR"
export RUN_DIR="$RUN_DIR/worker-$KUBEBRAIN_REPLICA"
export NORTHGATE_CAPTURE_MODULE=ops.northgate_capture.cp128k
exec bash ops/northgate_capture/retest.sh
