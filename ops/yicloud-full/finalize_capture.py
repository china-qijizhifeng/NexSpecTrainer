"""Publish a complete full-data feature view only after all four partitions pass."""
import json,os
from pathlib import Path
root=Path(os.environ['CAPTURE_RUN']);migration=Path(os.environ['MIGRATION_ROOT'])
plan=json.loads((migration/'full-data/inputs-manifest.json').read_text());windows=[]
for rank in range(4):
 work=root/f'capture-w{rank}';result=json.loads((work/'RESULT.json').read_text())
 assert result['passed'] and result['complete_partition'] and result['worker']==rank
 windows.extend(json.loads((work/'windows-manifest.json').read_text()))
assert len({x['id'] for x in windows})==len(windows)==plan['train_windows']+plan['validation_windows']
for split in ['train','validation']:
 out=root/(split+'-features');out.mkdir(exist_ok=True)
 for w in windows:
  if w['split']!=split:continue
  source=Path(w['file']);assert source.stat().st_size==w['bytes']
  target=out/w['id']
  if not target.exists():target.symlink_to(source)
  assert target.resolve()==source.resolve()
original=sorted([w for w in windows if w['split']=='train'],key=lambda w:w['id']);duplicates=[]
for index in range(plan['padding_duplicates']):
 w=original[index];target=root/'train-features'/f'padding-{index:04d}.ckpt.zst'
 if not target.exists():target.symlink_to(w['id'])
 assert target.resolve()==Path(w['file']).resolve()
 duplicates.append(dict(id=target.name,source=w['id'],input_token_sha256=w['input_token_sha256']))
assert len(list((root/'train-features').glob('*.ckpt.zst')))==plan['padded_train_windows']
assert len(list((root/'validation-features').glob('*.ckpt.zst')))==plan['validation_windows']
(root/'windows-manifest.json').write_text(json.dumps(dict(windows=windows,duplicates=duplicates)))
result=dict(passed=True,complete=True,counts=plan['counts'],total_steps=plan['total_steps'],full_training_split_captured=True,full_validation_split_captured=True,train_windows=plan['padded_train_windows'],validation_windows=plan['validation_windows'])
tmp=root/'capture.complete.partial';tmp.write_text(json.dumps(result,indent=2));tmp.replace(root/'capture.complete.json');print(json.dumps(result))
