#!/bin/bash
set -euo pipefail
: "${RUN_DIR:?}" "${KUBEBRAIN_REPLICA:?}"
export NORTHGATE_RESIDENT_RUN_DIR="$RUN_DIR"
export RUN_DIR="$RUN_DIR/bootstrap-$KUBEBRAIN_REPLICA"
export NORTHGATE_CAPTURE_MODULE=ops.northgate_capture.resident
exec bash ops/northgate_capture/retest.sh
