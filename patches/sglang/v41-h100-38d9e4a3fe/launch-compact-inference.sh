#!/usr/bin/env bash
set -euo pipefail
# Requires pinned SGLang image + source overlay.
# Compact inference; full hidden capture requires separate configuration.
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_OPT_DSV41_SM90_INDEXER=1
export SGLANG_OPT_DSV41_SM90_ATTN_MHC_ALL_REDUCE=1
export SGLANG_OPT_DSV41_SM90_DEEPSELECT_PREFILL=1
export SGLANG_OPT_DSV41_SM90_PREFILL_IO=1
export SGLANG_OPT_DSV41_SM90_PREFILL_MHC=1
exec sglang serve --model-path /gpfs/users/linlizhi/models/huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017 --trust-remote-code --tp-size 8 --ep-size 8 --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4 --engram-config '{"placement":"cpu"}' --mem-fraction-static 0.8 --context-length 69632 --max-total-tokens 131072 --swa-full-tokens-ratio 0.8 --max-prefill-tokens 65536 --max-running-requests 1 --skip-tokenizer-init --chunked-prefill-size 16384 --cuda-graph-backend-prefill breakable --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 --host 127.0.0.1 --port 30000 --random-seed 20261105 --decode-log-interval 1 --cuda-graph-bs-prefill 16384 --enable-nccl-nvls --enable-decoder-swa-bounded-replay
