"""Freeze one full trajectory and validate the native capture contract."""
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
import yaml
from scripts.prepare_server_hidden_states import check_capacity, iter_tasks
from specforge.application import resolve_offline_capture
from specforge.config import Config
from specforge.inference.capture import CaptureConfig
from specforge.training.capture_contract import resolve_server_capture_contract

root = Path(os.environ['RUN_DIR'])
repo = Path(os.environ['REPO_ROOT'])
migration = Path(os.environ['MIGRATION_ROOT'])
model = Path('/gpfs/shared-model-weight/models/deepseek-ai/DeepSeek-V4.1-Flash-1017')
revision = '1bae0e1088993b9b82de08f59de5be5a34e42170'
assert shutil.disk_usage(root).free > 10 * 1024**3
manifest = json.loads((repo/'ops/yicloud-capture/model-files.json').read_text())
assert manifest['revision'] == revision
for entry in manifest['files']:
    if entry['path'] == '.gitattributes':
        continue  # ModelScope packaging metadata differs, not model contents.
    path = model/entry['path']
    assert path.stat().st_size == entry['bytes'], entry['path']
    if not entry['path'].endswith('.safetensors'):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], entry['path']
source = migration/'inputs/train-full.jsonl'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '4add96ff71e7e8198b6263a4ea9b7c6f6b09c080475201ba0c84e1c113167d8e'
row = next(x for x in map(json.loads, source.read_text().splitlines())
           if x['id'] == 'denovo-a0fd0e29030cad04c8df2bff')
assert len(row['input_ids']) == len(row['loss_mask']) == 11590
assert row['metadata']['model_revision'] == revision
for name, digest, dtype in [('input_ids','token_sha256',torch.int64), ('loss_mask','mask_sha256',torch.uint8)]:
    value = torch.tensor(row[name], dtype=dtype).numpy().tobytes()
    assert hashlib.sha256(value).hexdigest() == row['metadata'][digest]
(root/'samples.jsonl').write_text(json.dumps(row)+'\n')
cfg = yaml.safe_load((repo/'ops/yicloud-capture/capture.yaml').read_text())
cfg['model'].update(target_model_path=str(model), draft_model_config=str(repo/'configs/deepseek-v41-flash-native.json'))
cfg['data'].update(hidden_states_path=str(root/'features'), cache_dir=str(root/'cache/data'))
cfg.update(run_id='yicloud-full-trajectory-capture', output_dir=str(root/'unused-training'))
(root/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
parsed = Config.from_file(str(root/'config.yaml'))
plan = resolve_offline_capture(parsed)
contract = resolve_server_capture_contract(parsed, algorithm=plan.run.algorithm)
assert plan.capture_layers == contract.aux_layer_ids == (37,38,39)
assert contract.target_hidden_size == 5120
capture = CaptureConfig.from_strategy(plan.layout.output_names, contract.aux_layer_ids,
    target_repr='hidden_state', target_hidden_size=contract.target_hidden_size,
    target_vocab_size=contract.target_vocab_size, draft_vocab_size=contract.draft_vocab_size)
tasks = list(iter_tasks(root/'samples.jsonl', cfg=parsed, run_id='preflight',
    target_version=revision, loss_mask_filter=plan.loss_mask_filter))
assert len(tasks) == 1
size = check_capacity(tasks[0].max_length, capture=capture, layout=plan.layout,
    client_bytes=8<<30, server_bytes=8<<30)
result = {'passed':True, 'sample':row['id'], 'tokens':11590, 'decoded_bytes':size,
    'layers':[37,38,39], 'target_revision':revision,
    'scope':'complete trajectory, one unchunked prefill; no truncation',
    'model_check':'all expected sizes and small-file SHA; previous structural check and first two weight SHA reused, not a new full-weight SHA audit'}
(root/'config-preflight.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
