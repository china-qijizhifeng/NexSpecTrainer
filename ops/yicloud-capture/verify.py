"""Read newly captured Zstd features through the trainer's native reader."""
import hashlib
import json
import os
from pathlib import Path
import torch
from specforge.data.feature_file import load_feature_file
from specforge.algorithms.common.hidden_states_data import normalize_dspark_offline_sample

root = Path(os.environ['RUN_DIR'])
row = json.loads((root/'samples.jsonl').read_text())
folder = root/'features'
path = folder/(row['id']+'.ckpt.zst')
assert len(list(folder.glob('*.ckpt.zst'))) == 1
assert not list(folder.glob('*.partial')) and not (folder/'.capture-writer.lock').exists()
plan = json.loads((folder/'capture-plan.json').read_text())
assert plan['aux_layer_ids'] == [37,38,39]
assert plan['target_model_version'] == row['metadata']['model_revision']
record = load_feature_file(path)
n = len(row['input_ids'])
for name in ('input_ids','loss_mask'):
    assert record[name].dtype == torch.int64
    assert torch.equal(record[name].reshape(-1),torch.tensor(row[name]))
for name,width in [('hidden_states',15360),('target_last_hidden_states',5120)]:
    tensor = record[name]
    assert tensor.shape == (1,n,width) and tensor.dtype == torch.bfloat16
    assert torch.isfinite(tensor).all() and tensor.abs().max() > 0
normalized = normalize_dspark_offline_sample(record,n)
assert normalized['hidden_states'].shape == (1,n,15360)
with path.open('rb') as stream:
    digest = hashlib.file_digest(stream,'sha256').hexdigest()
result = {'passed':True,'sample':row['id'],'tokens':n,'aux_layers':[37,38,39],
    'aux_shape':[1,n,15360],'final_shape':[1,n,5120],'dtype':'bfloat16',
    'tokens_and_mask_exact':True,'finite':True,'native_reader_passed':True,
    'file':str(path),'file_bytes':path.stat().st_size,'sha256':digest,
    'scope':'one real complete DeNovoSWE trajectory captured on YiCloud; not full dataset or training acceptance'}
(root/'RESULT.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
