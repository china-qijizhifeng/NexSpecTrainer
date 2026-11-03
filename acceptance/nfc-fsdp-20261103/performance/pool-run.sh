#!/bin/bash
set -euo pipefail
DSPARK_PERF_ROOT=/gpfs/users/linlizhi/nexspec-nfc-dspark-perf-20261103
mkdir -p "$DSPARK_PERF_ROOT/evidence/r4"
exec > >(tee "$DSPARK_PERF_ROOT/evidence/r4/worker.log") 2>&1
trap 'code=$?; echo "$code" > "$DSPARK_PERF_ROOT/evidence/r4/exit.txt"' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 -m pip install --no-index --no-deps --target /tmp/nfc-dspark-deps /gpfs/users/linlizhi/nexspec-v41-rdma-realtrace-20261103/wheels/*.whl
export PYTHONPATH="/gpfs/users/linlizhi/nexspec-nfc-native-20261103/source-r4:/gpfs/users/linlizhi/nfc-sharded-pipeline-20261103/source/src:/tmp/nfc-dspark-deps:${PYTHONPATH:-}"
cd /gpfs/users/linlizhi/nexspec-nfc-native-20261103/source-r4
python3 - <<'ENV'
import json, sys, socket, inspect, hashlib, importlib.metadata as md, torch, google.protobuf
from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
from megatron.core.dist_checkpointing import serialization
from google.protobuf.internal.api_implementation import Type
from pathlib import Path
assert 'async_strategy' in inspect.signature(TorchDistSaveShardedStrategy._get_save_and_finalize_callbacks).parameters
x=dict(core_file=serialization.__file__, core_sha256=hashlib.sha256(Path(serialization.__file__).read_bytes()).hexdigest(), python=sys.version, torch=torch.__version__, host=socket.gethostname(), te=md.version('transformer-engine'), protobuf=google.protobuf.__version__, protobuf_backend=Type(), gpu_count=torch.cuda.device_count(), gpu=torch.cuda.get_device_name(0))
Path('/gpfs/users/linlizhi/nexspec-nfc-dspark-perf-20261103/evidence/r4/environment.json').write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps(x),flush=True)
assert x['gpu_count'] == 8 and 'H100' in x['gpu']
ENV
python3 -m torch.distributed.run --standalone --nproc_per_node=8 "$DSPARK_PERF_ROOT/nfc_dspark_perf.py" --output "$DSPARK_PERF_ROOT/evidence/r4/results"
