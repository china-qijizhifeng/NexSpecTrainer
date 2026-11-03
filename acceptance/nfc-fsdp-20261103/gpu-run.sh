#!/bin/bash
set -euo pipefail
NATIVE_TEST_ROOT=/gpfs/users/linlizhi/nexspec-nfc-native-20261103
mkdir -p "$NATIVE_TEST_ROOT/evidence/r5"
exec > >(tee "$NATIVE_TEST_ROOT/evidence/r5/worker.log") 2>&1
trap 'code=$?; echo "$code" > "$NATIVE_TEST_ROOT/evidence/r5/exit.txt"' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 -m pip install --no-index --no-deps --target /tmp/nfc-native-deps /gpfs/users/linlizhi/nexspec-v41-rdma-realtrace-20261103/wheels/*.whl
export PYTHONPATH="$NATIVE_TEST_ROOT/source-r4:/gpfs/users/linlizhi/nfc-sharded-pipeline-20261103/source/src:/tmp/nfc-native-deps:${PYTHONPATH:-}"
cd "$NATIVE_TEST_ROOT/source-r4"
python3 - <<'ENV'
import json, sys, socket, inspect, hashlib, importlib.metadata as md, torch, google.protobuf
from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync
from megatron.core.dist_checkpointing import serialization
from specforge.training.nfc_checkpoint import NativeCheckpoint
from specforge.training.nfc_state import NativeState
from google.protobuf.internal.api_implementation import Type
from pathlib import Path
assert 'async_strategy' in inspect.signature(TorchDistSaveShardedStrategy._get_save_and_finalize_callbacks).parameters
x=dict(core_file=serialization.__file__, core_sha256=hashlib.sha256(Path(serialization.__file__).read_bytes()).hexdigest(), python=sys.version, torch=torch.__version__, host=socket.gethostname(), te=md.version('transformer-engine'), protobuf=google.protobuf.__version__, protobuf_backend=Type(), gpu_count=torch.cuda.device_count(), gpu=torch.cuda.get_device_name(0))
Path('/gpfs/users/linlizhi/nexspec-nfc-native-20261103/evidence/r5/environment.json').write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps(x),flush=True)
assert x['gpu_count'] == 8
ENV
python3 -m torch.distributed.run --standalone --nproc_per_node=8 tests/gpu/nfc_fsdp_acceptance.py --output "$NATIVE_TEST_ROOT/evidence/r5/results"
