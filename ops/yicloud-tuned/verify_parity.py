"""Compare every selected aux/final/token/mask tensor to the original captured run."""
import json,os
from pathlib import Path
import torch
from specforge.data.feature_file import load_feature_file
work=Path(os.environ['RUN_DIR']);old=Path(os.environ['PREVIOUS_CAPTURE']);rank=int(os.environ['KUBEBRAIN_REPLICA'])
row=json.loads((work/'parity-sample.jsonl').read_text());sample=row['id']
new=json.loads((work/'parity/sliced'/f'{sample}.json').read_text());prior=json.loads((old/f'capture-w{rank}/sliced'/f'{sample}.json').read_text())
assert new['complete'] and prior['complete'] and new['input']==prior['input']
assert len(new['windows'])==len(prior['windows'])
for a,b in zip(new['windows'],prior['windows']):
 assert a['id']==b['id']
 x=load_feature_file(a['file']);y=load_feature_file(b['file']);assert x.keys()==y.keys()
 for key in x:
  assert torch.equal(x[key],y[key]),f"fused indexer full-output mismatch: {a['id']} {key}"
result=dict(passed=True,worker=rank,id=sample,tokens=len(row['input_ids']),windows=len(new['windows']),all_tensors_bitwise_equal=True)
(work/'parity/RESULT.json').write_text(json.dumps(result));print(json.dumps(result),flush=True)
