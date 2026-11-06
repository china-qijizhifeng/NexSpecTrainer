#!/bin/bash
set -euo pipefail
: "${RUN_DIR:?}" "${INPUT_ROOT:?}" "${MIGRATION_ROOT:?}" "${KUBEBRAIN_REPLICA:?}" "${KUBEBRAIN_REPLICA_TOTAL:?}" "${KUBEBRAIN_HOSTNAME_PREFIX:?}" "${YICLOUD_RENDEZVOUS_NAMESPACE:?}"
export REPO_ROOT="$PWD" TASK_ROOT="$RUN_DIR"
export NODE_ROOT="$RUN_DIR/nodes/$KUBEBRAIN_REPLICA"
mkdir -p "$NODE_ROOT/training" "$RUN_DIR/training" "$RUN_DIR/gpu-regression"
exec > >(tee "$NODE_ROOT/training/worker.log") 2>&1
trap 'code=$?; echo "$code" > "$NODE_ROOT/training/exit.txt"' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_NO_SYNC=1
export PYTHONPATH="$REPO_ROOT:$MIGRATION_ROOT/runtime-deps/train-core:$MIGRATION_ROOT/runtime-deps/nfc-src:/opt/nexspec/overlay/nvidia_cutlass_dsl/dsl_packages:/opt/nexspec/overlay:/opt/nexspec/tile:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="$NODE_ROOT/cache/triton" CUDA_CACHE_PATH="$NODE_ROOT/cache/cuda" XDG_CACHE_HOME="$NODE_ROOT/cache/xdg"
export DG_JIT_CACHE_DIR="$NODE_ROOT/cache/deepgemm" TILELANG_CACHE_DIR="$NODE_ROOT/cache/tilelang"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$DG_JIT_CACHE_DIR" "$TILELANG_CACHE_DIR"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_DEBUG=INFO
export NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0
: "${NCCL_IB_HCA:?training runtime must select active IB ports}"
export NCCL_DEBUG_FILE="$NODE_ROOT/training/nccl-%h-%p.log"
export TORCH_FR_BUFFER_SIZE=20000 TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TRAIN_MASTER_ADDR="${KUBEBRAIN_HOSTNAME_PREFIX}-0.${YICLOUD_RENDEZVOUS_NAMESPACE}.svc.lg.shzhisuan.local"
echo '[stage] environment and native input checks'
python3 ops/yicloud-adamw32/preflight.py --gpu
python3 ops/yicloud-adamw32/network.py
source "$NODE_ROOT/network.env"
python3 ops/yicloud-adamw32/prepare_node.py
echo '[stage] 32-rank AdamW streamed FSDP gate'
launch=(python3 -m torch.distributed.run --nnodes=4 --nproc_per_node=8 --node_rank="$KUBEBRAIN_REPLICA" --master_addr="$TRAIN_MASTER_ADDR" --max_restarts=0)
"${launch[@]}" --master_port=29772 ops/yicloud-adamw32/fixed_gate.py --output "$RUN_DIR/gpu-regression"
echo '[stage] native all-anchor AdamW training'
"${launch[@]}" --master_port=29773 ops/yicloud-adamw32/audit_train.py train --config "$NODE_ROOT/train.yaml"
if [ "$KUBEBRAIN_REPLICA" = 0 ]; then
 python3 ops/yicloud-adamw32/finalize.py
fi
