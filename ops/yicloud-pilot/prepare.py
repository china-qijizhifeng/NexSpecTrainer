"""Freeze a three-update real-feature pilot and verify its exact input records."""
import hashlib,json,os,shutil
from pathlib import Path
import torch,yaml
from specforge.data.feature_file import load_feature_file
from specforge.algorithms.common.hidden_states_data import normalize_dspark_offline_sample
root=Path(os.environ['RUN_DIR']);migration=Path(os.environ['MIGRATION_ROOT']);repo=Path(os.environ['REPO_ROOT'])
model=Path('/gpfs/shared-model-weight/models/deepseek-ai/DeepSeek-V4.1-Flash-1017')
assert shutil.disk_usage(root).free>850_000_000_000
selected=json.loads((migration/'pilot-selected-windows.json').read_text())
assert len(selected)==9 and sum(x['split']=='train' for x in selected)==8
report=[]
for split in ('train','validation'):
 rows={r['id']:r for r in map(json.loads,(migration/'inputs'/f'{split}-full.jsonl').read_text().splitlines())}
 for item in selected:
  if item['split']!=split:continue
  p=migration/'pilot-features'/split/(item['id']+'.ckpt.zst')
  assert p.stat().st_size==item['bytes']
  with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
  data=load_feature_file(p);normalize_dspark_offline_sample(data,4096)
  row=rows[item['id'].rsplit('-s',1)[0]];start,end=item['training_slice']
  assert start==0 and end==4096
  assert torch.equal(data['input_ids'].reshape(-1),torch.tensor(row['input_ids'][start:end]))
  assert torch.equal(data['loss_mask'].reshape(-1),torch.tensor(row['loss_mask'][start:end]))
  for key,dim in [('hidden_states',15360),('target_last_hidden_states',5120)]:
   assert data[key].shape==(1,4096,dim) and data[key].dtype==torch.bfloat16
   assert torch.isfinite(data[key]).all()
  assert ((data['loss_mask'].reshape(-1)[:-1]>0)&(data['loss_mask'].reshape(-1)[1:]>0)).any()
  report.append({'path':str(p),'sha256':item['sha256'],'target_context':item['target_context'],'training_slice':[start,end]})
config=yaml.safe_load((migration/'baseline/train.yaml').read_text())
config['model'].update(target_model_path=str(model),draft_model_config=str(repo/'configs/deepseek-v41-flash-native.json'))
config['data'].update(hidden_states_path=str(migration/'pilot-features/train'),eval_hidden_states_path=str(migration/'pilot-features/validation'),cache_dir=str(root/'cache/data'),dataloader_num_workers=0)
config['training'].update(accumulation_steps=1,num_epochs=4,max_steps=3,total_steps=3,save_interval=0,eval_interval=0,max_checkpoints=1)
config.update(run_id='yicloud-denovo-pilot',output_dir=str(root/'training'))
initial=json.loads(json.dumps(config));initial['training']['max_steps']=1
(root/'initial.yaml').write_text(yaml.safe_dump(initial,sort_keys=False))
config['training']['resume_from']=str(root/'training/yicloud-denovo-pilot-step1')
(root/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
(root/'input-check.json').write_text(json.dumps({'passed':True,'records':report,'scope':'one first 4K window per trajectory from complete-context BP capture; 8 train + 1 validation, validation not evaluated in this pilot'},indent=2))
print(json.dumps({'input_check':'passed','train_windows':8,'max_steps':3,'save_restore_at':1}),flush=True)
