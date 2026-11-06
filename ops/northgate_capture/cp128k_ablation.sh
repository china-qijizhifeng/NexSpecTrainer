#!/bin/bash
set -euo pipefail
export NORTHGATE_CAPTURE_RUN_DIR="$RUN_DIR"
export RUN_DIR="$RUN_DIR/worker-${KUBEBRAIN_REPLICA:-0}"
export NORTHGATE_CAPTURE_MODULE=ops.northgate_capture.cp128k_ablation
exec bash ops/northgate_capture/retest.sh
