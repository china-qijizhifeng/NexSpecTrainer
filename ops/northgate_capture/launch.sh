#!/bin/bash
set -euo pipefail
: "${RUN_DIR:?}" "${NORTHGATE_ROOT:?}" "${KUBEBRAIN_REPLICA:?}"
export RUN_ROOT="$RUN_DIR"
export RUN_DIR="$RUN_ROOT/worker-${KUBEBRAIN_REPLICA}"
export CAPTURE_WORK_DIR="$RUN_DIR"
export REPO_ROOT="$PWD"
mkdir -p "$RUN_DIR/python-deps" "$RUN_DIR/features"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export PYTHONPATH="$RUN_DIR/python-deps:$REPO_ROOT:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_NO_SYNC=1
export SPECFORGE_MAX_CAPTURE_RECORD_BYTES=34359738368
# Offline, run-private codec/plugin installation; preserve image Torch and TE.
python3 -m pip install --no-index --no-deps --target "$RUN_DIR/python-deps" "$NORTHGATE_ROOT"/wheels/*.whl
cp -a plugins/sglang-spec-capture "$RUN_DIR/plugin-source"
python3 -m pip install --no-index --no-deps --no-build-isolation --target "$RUN_DIR/python-deps" "$RUN_DIR/plugin-source"
python3 ops/northgate_capture/worker_prepare.py
python3 - <<'PY'
import importlib.metadata as md,json,os
from pathlib import Path
import torch,brotli,zstandard
from specforge.data.feature_file import require_hybrid_compression
require_hybrid_compression()
assert torch.cuda.device_count()==8
assert all('H200' in torch.cuda.get_device_name(i) for i in range(8))
root=Path(os.environ['RUN_DIR'])
root.joinpath('environment.json').write_text(json.dumps({'torch':torch.__version__,'brotli':brotli.__version__,'zstandard':zstandard.__version__,
 'sglang':md.version('sglang'),'gpus':[torch.cuda.get_device_name(i) for i in range(8)],
 'max_record_bytes':int(os.environ['SPECFORGE_MAX_CAPTURE_RECORD_BYTES'])},indent=2))
PY
CONTEXT=$(cat "$RUN_DIR/context.txt")
unset MC_FORCE_TCP MC_MS_AUTO_DISC MC_USE_TENT MC_USE_TEV1 MC_TE_FILTERS MOONCAKE_RDMA_DEVICES
export SPECFORGE_SPEC_CAPTURE=1 SGLANG_PLUGINS=specforge_spec_capture NORTHGATE_LONG_CAPTURE=1
export MOONCAKE_PROTOCOL=tcp MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551
export MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata
export MOONCAKE_GLOBAL_SEGMENT_SIZE=34359738368 MOONCAKE_LOCAL_BUFFER_SIZE=34359738368
export SGLANG_SPEC_CAPTURE_GPU_PUT=0 MC_STORE_MEMCPY=0 MC_TRANSFER_TIMEOUT=600
export SGLANG_SPEC_CAPTURE_WIRE_ENCODING=raw SPECFORGE_ZSTD_WORKERS=8
export TRITON_CACHE_DIR="$RUN_DIR/cache/triton" CUDA_CACHE_PATH="$RUN_DIR/cache/cuda"
export XDG_CACHE_HOME="$RUN_DIR/cache/xdg" SGLANG_CACHE_DIR="$RUN_DIR/cache/sglang"
export DG_JIT_CACHE_DIR="$RUN_DIR/cache/deepgemm" TILELANG_CACHE_DIR="$RUN_DIR/cache/tilelang"
export FLASHINFER_WORKSPACE_BASE="$RUN_DIR/cache/flashinfer"
export PYTORCH_ALLOC_CONF=expandable_segments:True SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$SGLANG_CACHE_DIR" "$DG_JIT_CACHE_DIR" "$TILELANG_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"
if [[ -n "${NORTHGATE_CP_REFERENCE_RUN:-}" ]]; then
  # Same pinned image/model/GPU architecture; clone rather than modify evidence.
  cp -a "$NORTHGATE_CP_REFERENCE_RUN/worker-${KUBEBRAIN_REPLICA}/cache/." "$RUN_DIR/cache/"
fi
ulimit -l unlimited
MASTER_PID= SERVER_PID=
cleanup() {
  code=$?
  trap - EXIT
  for pid in "$SERVER_PID" "$MASTER_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then kill -- "-$pid" || true; fi
  done
  echo "$code" > "$RUN_DIR/exit.txt"
  exit "$code"
}
trap cleanup EXIT
setsid mooncake_master --enable_http_metadata_server=true --http_metadata_server_host=127.0.0.1 \
 --rpc_port=35551 --http_metadata_server_port=35880 --metrics_port=35900 \
 --default_kv_lease_ttl=30m > "$RUN_DIR/mooncake.log" 2>&1 &
MASTER_PID=$!
setsid sglang serve \
 --model-path /gpfs/shared-model-weight/models/deepseek-ai/DeepSeek-V4.1-Flash-1017 --trust-remote-code \
 --language-model-only --enable-prefill-cp --cp-strategy interleave \
 --tp-size 8 --ep-size 8 --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4 \
 --engram-config '{"placement":"cpu"}' \
 --mem-fraction-static 0.8 --context-length "$CONTEXT" --max-total-tokens "$CONTEXT" \
 --swa-full-tokens-ratio 1.0 --max-prefill-tokens "$CONTEXT" --max-running-requests 1 \
 --skip-tokenizer-init --chunked-prefill-size -1 --disable-radix-cache \
 --speculative-algorithm DSPARK --enable-return-aux-hidden-states \
 --return-aux-hidden-state-layer-ids 37 38 39 --return-hidden-states-mode full \
 --no-enable-decoder-swa-bounded-replay \
 --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled \
 --host 127.0.0.1 --port 30000 > "$RUN_DIR/serve.log" 2>&1 &
SERVER_PID=$!
python3 ops/northgate_capture/wait_server.py --pid "$SERVER_PID" --output "$RUN_DIR" --timeout 5400 --capture --cp-size 8
if [[ "$KUBEBRAIN_REPLICA" != 0 ]]; then
  python3 ops/northgate_capture/cp_acceptance.py --wait "$RUN_ROOT"
fi
python3 ops/northgate_capture/collect.py --config "$RUN_DIR/config.yaml" \
 --server http://127.0.0.1:30000 --samples "$RUN_DIR/samples.jsonl" --output "$RUN_DIR/features" \
 --target-model-version 1bae0e1088993b9b82de08f59de5be5a34e42170 \
 --server-capture plugin --client-buffer-bytes 34359738368 --server-buffer-bytes 34359738368 \
 --wire-encoding raw --pipeline-depth 1 --hybrid --brotli-quality 10 --compression-workers 24 \
 --timeout-s 7200 > "$RUN_DIR/client.log" 2>&1
python3 - <<'PY'
import json,os
from pathlib import Path
root=Path(os.environ['RUN_DIR']);expected=sum(1 for _ in (root/'samples.jsonl').open())
receipts=list((root/'receipts').glob('*.json'))
assert len(receipts)==expected
result={'passed':True,'records':expected,'all_files_hybrid':True,'brotli_quality':10}
(root/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
PY
