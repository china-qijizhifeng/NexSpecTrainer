"""Account for every trajectory and every derived window before completion."""
import json,os
from pathlib import Path
root=Path(os.environ['RUN_DIR']);migration=Path(os.environ['MIGRATION_ROOT'])
plan=json.loads((migration/'full-data/inputs-manifest.json').read_text())
rank=int(os.environ['KUBEBRAIN_REPLICA']);samples=[x for x in plan['samples'] if x['worker']==rank]
windows=[]
for sample in samples:
 r=json.loads((root/'sliced'/(sample['id']+'.json')).read_text())
 assert r['complete'] and r['input']['token_sha256']==sample['token_sha256']
 assert r['input']['mask_sha256']==sample['mask_sha256']
 assert len(r['windows'])==len(sample['slice_starts'])
 for w,start in zip(r['windows'],sample['slice_starts']):
  assert w['codec_bit_exact'] and w['training_slice']==[start,min(start+4096,sample['num_tokens'])]
  assert w['target_context']==[0,sample['num_tokens']] and w['split']==sample['split']
  assert Path(w['file']).stat().st_size==w['bytes']
  windows.append(w)
assert len(list((root/'sliced').glob('*.json')))==len(samples)
for split in ['train','validation']:
 assert len(list((root/(split+'-features')).glob('*.ckpt.zst')))==sum(w['split']==split for w in windows)
assert not list((root/'features').glob('*.ckpt.zst'))
(root/'windows-manifest.json').write_text(json.dumps(windows,indent=2))
(root/'RESULT.json').write_text(json.dumps(dict(passed=True,worker=rank,trajectories=len(samples),windows=len(windows),complete_partition=True,all_codec_bit_exact=True)))
