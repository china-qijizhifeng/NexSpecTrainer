"""Freeze the full epoch and validate one native record from each split."""
import hashlib,json,os,shutil
from pathlib import Path
import torch,yaml
from specforge.data.feature_file import load_feature_file
from specforge.algorithms.common.hidden_states_data import normalize_dspark_offline_sample
from specforge.config.schema import Config
root=Path(os.environ['RUN_DIR']);migration=Path(os.environ['MIGRATION_ROOT']);repo=Path(os.environ['REPO_ROOT']);capture=Path(os.environ['CAPTURE_RUN'])
assert shutil.disk_usage(root).free>850_000_000_000
plan=json.loads((migration/'full-data/inputs-manifest.json').read_text());done=json.loads((capture/'capture.complete.json').read_text())
assert done['passed'] and done['counts']==plan['counts'] and done['total_steps']==796
windows=json.loads((capture/'windows-manifest.json').read_text())
for split,count in [('train',101888),('validation',5896)]:
 paths=sorted((capture/(split+'-features')).glob('*.ckpt.zst'));assert len(paths)==count
 p=paths[0];w=next(w for w in windows['windows'] if w['id']==p.name)
 with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==w['sha256']
 data=load_feature_file(p);normalize_dspark_offline_sample(data,4096)
 for key,width in [('hidden_states',15360),('target_last_hidden_states',5120)]:
  assert data[key].shape[-1]==width and data[key].dtype==torch.bfloat16 and torch.isfinite(data[key]).all()
config=yaml.safe_load((migration/'baseline/train.yaml').read_text())
config['model'].update(target_model_path='/gpfs/shared-model-weight/models/deepseek-ai/DeepSeek-V4.1-Flash-1017',draft_model_config=str(repo/'configs/deepseek-v41-flash-native.json'))
config['data'].update(hidden_states_path=str(capture/'train-features'),eval_hidden_states_path=str(capture/'validation-features'),cache_dir=str(root/'cache/data'),dataloader_num_workers=2)
config['training'].update(accumulation_steps=16,num_epochs=1,max_steps=796,total_steps=796,save_interval=100,eval_interval=100,max_checkpoints=1)
config.update(run_id='yicloud-denovo-full',output_dir=str(root/'training'))
(root/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False));parsed=Config.from_file(str(root/'config.yaml'));parsed.validate_world_size(8)
(root/'input-check.json').write_text(json.dumps(dict(passed=True,counts=plan['counts'],train_windows=101888,validation_windows=5896,optimizer_updates=796,full_epoch=True,explicit_padding_duplicates=16)))
