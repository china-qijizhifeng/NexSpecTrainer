"""Freeze full-data counts while admitting only checked, already captured files."""
import json
import os
import shutil
from pathlib import Path
import torch
import yaml
from specforge.data.feature_file import load_feature_file
from specforge.algorithms.common.hidden_states_data import normalize_dspark_offline_sample
from specforge.config.schema import Config
from schedule import atomic, sha256, validate_receipt, RESERVE, SEED, QUANTUM

root=Path(os.environ['RUN_DIR']); migration=Path(os.environ['MIGRATION_ROOT']); repo=Path(os.environ['REPO_ROOT']); capture=Path(os.environ['CAPTURE_RUN']); stream=Path(os.environ['STREAM_ROOT'])
assert shutil.disk_usage(root).free > 850_000_000_000
plan_path=migration/'full-data/inputs-manifest.json'; plan=json.loads(plan_path.read_text())
assert (plan['padded_train_windows'],plan['validation_windows'],plan['total_steps'])==(101888,5896,796)
assert json.loads((capture/'run.json').read_text())['uuid']=='05a1489a-0f70-46ad-a1d0-86cde069b664'
samples={s['id']:s for s in plan['samples']}; checked={}
for rank in range(4):
 for receipt in sorted((capture/f'capture-w{rank}'/'sliced').glob('*.json')):
  sample=samples[receipt.stem]
  if sample['split'] in checked: continue
  windows=validate_receipt(json.loads(receipt.read_text()),sample,capture); w=windows[0]; path=Path(w['file'])
  assert sha256(path)==w['sha256']
  data=load_feature_file(path); normalize_dspark_offline_sample(data,4096)
  for key,width in [('hidden_states',15360),('target_last_hidden_states',5120)]:
   assert data[key].shape[-1]==width and data[key].dtype==torch.bfloat16 and torch.isfinite(data[key]).all()
  checked[sample['split']]=w
assert set(checked)=={'train','validation'}
contract=dict(version=1,train_slots=101888,validation_slots=5896,quantum=QUANTUM,reserve=RESERVE,seed=SEED,inputs_sha256=sha256(plan_path),capture_uuid='05a1489a-0f70-46ad-a1d0-86cde069b664',stream_root=str(stream),ordering='arrival-dependent buffered shuffle; immutable blocks are replay authority; long-first capture is not global shuffle')
assert not (stream/'contract.json').exists()
atomic(stream/'contract.json',contract)
config=yaml.safe_load((migration/'baseline/train.yaml').read_text())
config['model'].update(target_model_path='/gpfs/shared-model-weight/models/deepseek-ai/DeepSeek-V4.1-Flash-1017',draft_model_config=str(repo/'configs/deepseek-v41-flash-native.json'))
config['data'].update(hidden_states_path=str(stream/'train'),eval_hidden_states_path=str(stream/'validation'),cache_dir=str(root/'cache/data'),dataloader_num_workers=2)
config['training'].update(accumulation_steps=16,num_epochs=1,max_steps=796,total_steps=796,save_interval=100,eval_interval=796,max_checkpoints=1)
config.update(run_id='yicloud-denovo-full-stream',output_dir=str(root/'training'))
(root/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False)); Config.from_file(str(root/'config.yaml')).validate_world_size(8)
atomic(root/'input-check.json',dict(passed=True,counts=plan['counts'],train_windows=101888,validation_windows=5896,optimizer_updates=796,full_epoch=True,explicit_padding_duplicates=16,checked_native_samples=checked,stream_contract=contract))
