#!/usr/bin/env bash
set -euo pipefail
cd "${WORKSPACE:?managed source worktree is required}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD:/opt/nexspec/overlay/nvidia_cutlass_dsl/dsl_packages:/opt/nexspec/overlay:/opt/nexspec/tile${PYTHONPATH:+:$PYTHONPATH}"
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK
python3 - <<'PY'
import hashlib, importlib.util, json, os, pathlib, subprocess, sys
import torch, triton
source = pathlib.Path(importlib.util.find_spec('sglang').origin).parent
files = ['srt/speculative/dspark_components/' + name + '.py' for name in
         ['dspark_worker_v2', 'dspark_config', 'dspark_planner', 'dspark_verify', 'dspark_deferred_commit']]
files += ['srt/managers/overlap_utils.py', 'srt/model_executor/model_runner.py',
          'srt/layers/attention/deepseek_v4_backend.py', 'srt/mem_cache/deepseek_v4_memory_pool.py',
          'srt/arg_groups/cuda_graph_hook.py',
          'srt/arg_groups/deepseek_v4_hook.py', 'srt/arg_groups/fields/spec.py']
result = dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
              cuda=torch.version.cuda, triton=triton.__version__, sglang_source=str(source),
              sglang_hashes={name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files},
              nvidia_smi=subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version',
                                                '--format=csv,noheader'], text=True))
pathlib.Path(os.environ['RUN_DIR'], 'kernel-environment.json').write_text(json.dumps(result, indent=2)+'\n')
PY
bundle="$WORKSPACE/patches/sglang/v41-hybrid-spark-38d9e4a3fe"
python3 -m specforge.hybrid.runtime_patch --root /sgl-workspace/sglang --bundle "$bundle" > "$RUN_DIR/runtime-patch-plan.json"
python3 -m specforge.hybrid.runtime_patch --root /sgl-workspace/sglang --bundle "$bundle" --execute > "$RUN_DIR/runtime-patch-result.json"
python3 - <<'PY'
import json, os, pathlib
from transformers import AutoConfig
from specforge.hybrid.config import HybridConfig
from sglang.srt.speculative.dspark_components.dspark_config import resolve_runtime_config
path = pathlib.Path(os.environ['RUN_DIR'], 'config-smoke.json')
path.write_text(json.dumps({'uncalibrated_acceptance':0.9})+'\n')
os.environ['SGLANG_DSPARK_HYBRID_CONFIG'] = str(path)
model = '/gpfs/shared-model-weight/models/deepseek-ai/DeepSeek-V4.1-Flash-1017'
config = AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=True)
runtime = resolve_runtime_config(draft_hf_config=config, speculative_num_draft_tokens=16,
                                target_vocab_size=config.vocab_size)
assert runtime.gamma == 5 and runtime.verify_num_draft_tokens == 16
pathlib.Path(os.environ['RUN_DIR'], 'horizon-resolution.json').write_text(json.dumps({
    'model_horizon':runtime.gamma, 'verify_capacity':runtime.verify_num_draft_tokens,
    'model_config_only':True, 'weights_loaded':False})+'\n')
PY
exec python3 tests/gpu/hybrid_spark_acceptance.py --output "${RUN_DIR:?}/kernel-results.json"
