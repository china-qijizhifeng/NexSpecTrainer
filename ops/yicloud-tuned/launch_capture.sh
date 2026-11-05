#!/bin/bash
set -euo pipefail
: "${RUN_DIR:?}" "${MIGRATION_ROOT:?}"
export RUN_ROOT="$RUN_DIR"
export RUN_DIR="$RUN_ROOT/capture-w${KUBEBRAIN_REPLICA}"
export CAPTURE_WORK_DIR="$RUN_DIR"
export SPECFORGE_MAX_CAPTURE_RECORD_BYTES=17179869184
export REPO_ROOT=$PWD PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_NO_SYNC=1
export PYTHONPATH="$REPO_ROOT/ops/yicloud-tuned:$RUN_DIR/python-deps:$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p "$RUN_DIR/python-deps" "$RUN_DIR/capture"
if ! python3 -c 'import zstandard' >/dev/null 2>&1; then
  python3 -m pip install --no-index --no-deps --target "$RUN_DIR/python-deps" "$MIGRATION_ROOT"/runtime-deps/wheels/*.whl
fi
cp -a "$REPO_ROOT/plugins/sglang-spec-capture" "$RUN_DIR/plugin-source"
python3 -m pip install --no-index --no-deps --no-build-isolation --target "$RUN_DIR/python-deps" "$RUN_DIR/plugin-source"
python3 ops/yicloud-tuned/prepare_capture.py
python3 ops/yicloud-tuned/test_slice.py > "$RUN_DIR/slice-regression.log" 2>&1
python3 ops/yicloud-tuned/validate_indexer.py --output "$RUN_DIR/indexer-regression.json" > "$RUN_DIR/indexer-regression.log" 2>&1
python3 - <<'CACHE'
import json,os,shutil
from pathlib import Path
probe=Path(os.environ['VERIFIED_PROBE']);root=Path(os.environ['RUN_DIR'])
result=json.loads((probe/'RESULT.json').read_text());assert result['passed'] and result['tokens']==234048
shutil.copytree(probe/'cache',root/'cache',dirs_exist_ok=True)
(root/'cache-reuse.json').write_text(json.dumps({'source':str(probe),'source_result':result,'mode':'private copy after terminal probe'}))
CACHE
if [[ ${CAPTURE_CPU_ONLY:-0} == 1 ]]; then
  python3 ops/yicloud-capture/probe_server_image.py --cpu > "$RUN_DIR/environment.json"
  exit 0
fi
python3 ops/yicloud-capture/probe_server_image.py > "$RUN_DIR/environment.json"
unset MC_FORCE_TCP MC_MS_AUTO_DISC MC_USE_TENT MC_USE_TEV1 MC_TE_FILTERS MOONCAKE_RDMA_DEVICES
export SPECFORGE_SPEC_CAPTURE=1 SGLANG_PLUGINS=specforge_spec_capture
export SPECFORGE_MHC_INDEX64=1 SPECFORGE_FUSED_INDEXER=1
export SPECFORGE_MOE_CHUNK_TOKENS=8192
export MOONCAKE_PROTOCOL=tcp MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551
export MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata
export MOONCAKE_GLOBAL_SEGMENT_SIZE=17179869184 MOONCAKE_LOCAL_BUFFER_SIZE=17179869184
export SGLANG_SPEC_CAPTURE_GPU_PUT=0 MC_STORE_MEMCPY=0 MC_TRANSFER_TIMEOUT=600
export SGLANG_SPEC_CAPTURE_WIRE_ENCODING=bf16_fields_zstd_v2 SPECFORGE_ZSTD_WORKERS=8
export TRITON_CACHE_DIR="$RUN_DIR/cache/triton" CUDA_CACHE_PATH="$RUN_DIR/cache/cuda"
export XDG_CACHE_HOME="$RUN_DIR/cache/xdg" SGLANG_CACHE_DIR="$RUN_DIR/cache/sglang"
export DG_JIT_CACHE_DIR="$RUN_DIR/cache/deepgemm" TILELANG_CACHE_DIR="$RUN_DIR/cache/tilelang"
export FLASHINFER_WORKSPACE_BASE="$RUN_DIR/cache/flashinfer"
export PYTORCH_ALLOC_CONF=expandable_segments:True SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$SGLANG_CACHE_DIR" "$DG_JIT_CACHE_DIR" "$TILELANG_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"
ulimit -l unlimited
MASTER_PID= SERVER_PID=
cleanup() {
  code=$?
  trap - EXIT
  for pid in "$SERVER_PID" "$MASTER_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then kill -- "-$pid" || true; fi
  done
  echo "$code" > "$RUN_DIR/capture/exit.txt"
  exit "$code"
}
trap cleanup EXIT
setsid mooncake_master --enable_http_metadata_server=true --http_metadata_server_host=127.0.0.1 \
  --rpc_port=35551 --http_metadata_server_port=35880 --metrics_port=35900 \
  --default_kv_lease_ttl=30m > "$RUN_DIR/capture/mooncake.log" 2>&1 &
MASTER_PID=$!
setsid sglang serve \
  --model-path /gpfs/shared-model-weight/models/deepseek-ai/DeepSeek-V4.1-Flash-1017 --trust-remote-code \
  --tp-size 8 --ep-size 8 --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4 \
  --engram-config '{"placement":"cpu"}' \
  --mem-fraction-static 0.8 --context-length 262144 --max-total-tokens 262144 \
  --swa-full-tokens-ratio 1.0 --max-prefill-tokens 262144 --max-running-requests 1 \
  --skip-tokenizer-init --chunked-prefill-size -1 --disable-radix-cache \
  --speculative-algorithm DSPARK --enable-return-aux-hidden-states \
  --return-aux-hidden-state-layer-ids 37 38 39 --return-hidden-states-mode full \
  --no-enable-decoder-swa-bounded-replay \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled \
  --host 127.0.0.1 --port 30000 > "$RUN_DIR/capture/serve.log" 2>&1 &
SERVER_PID=$!
python3 ops/yicloud-capture/wait_server.py --pid "$SERVER_PID" --output "$RUN_DIR/capture" --timeout 5400 --capture
python3 ops/yicloud-full/validate_server_pool.py
mkdir -p "$RUN_DIR/parity"
CAPTURE_WORK_DIR="$RUN_DIR/parity" python3 ops/yicloud-tuned/capture_and_slice.py --config "$RUN_DIR/config.yaml" \
  --server http://127.0.0.1:30000 --samples "$RUN_DIR/parity-sample.jsonl" --output "$RUN_DIR/parity/features" \
  --target-model-version 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --server-capture plugin --client-buffer-bytes 17179869184 --server-buffer-bytes 17179869184 \
  --wire-encoding bf16_fields_zstd_v2 --zstd --zstd-level 1 --timeout-s 7200 > "$RUN_DIR/parity/client.log" 2>&1
python3 ops/yicloud-tuned/verify_parity.py
python3 ops/yicloud-tuned/capture_and_slice.py --config "$RUN_DIR/config.yaml" \
  --server http://127.0.0.1:30000 --samples "$RUN_DIR/samples.jsonl" --output "$RUN_DIR/features" \
  --target-model-version 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --server-capture plugin --client-buffer-bytes 17179869184 --server-buffer-bytes 17179869184 \
  --wire-encoding bf16_fields_zstd_v2 --pipeline-depth 2 --zstd --zstd-level 1 --timeout-s 7200 > "$RUN_DIR/capture/client.log" 2>&1
python3 ops/yicloud-full/verify_partition.py
