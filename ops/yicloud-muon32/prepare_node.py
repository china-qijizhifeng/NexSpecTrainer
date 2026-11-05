"""Validate the selected inputs and materialize this node's resolved recipe."""
import hashlib,json,os,shutil
from pathlib import Path
import torch,yaml
from specforge.config import load_config
from specforge.data.feature_file import load_feature_file
from specforge.algorithms.common.hidden_states_data import normalize_dspark_offline_sample
root=Path(os.environ['RUN_DIR']);node=Path(os.environ['NODE_ROOT']);inputs=Path(os.environ['INPUT_ROOT']);ops=Path(__file__).parent
assert int(os.environ['KUBEBRAIN_REPLICA_TOTAL'])==4
rank=int(os.environ['KUBEBRAIN_REPLICA'])
assert 0<=rank<4
assert shutil.disk_usage('/dev/shm').free>=288*2**30
receipt=json.loads((inputs/'input-compatibility.json').read_text());assert receipt['passed']
assert receipt['train_traces']==1326 and receipt['validation_traces']==17
assert len(list((inputs/'train-features').glob('*.ckpt.zst')))==15104
assert len(list((inputs/'validation-features').glob('*.ckpt.zst')))==136
master=os.environ['TRAIN_MASTER_ADDR']
expected={v['input_sha256']:v for v in json.loads((inputs/'data-audit.json').read_text())['windows'].values()};checked=[]
for split in ['train','validation']:
 path=sorted((inputs/f'{split}-features').glob('denovo*.ckpt.zst'))[rank]
 raw=load_feature_file(str(path));normalize_dspark_offline_sample(raw,4096)
 h=hashlib.sha256(raw['input_ids'].contiguous().view(torch.uint8).numpy()).hexdigest();e=expected[h]
 assert e['split']==split
 assert hashlib.sha256((raw['loss_mask']>.5).numpy().astype('u1').tobytes()).hexdigest()==e['mask_sha256']
 for k,w in [('hidden_states',15360),('target_last_hidden_states',5120)]:
  assert raw[k].shape[-1]==w and raw[k].dtype==torch.bfloat16 and torch.isfinite(raw[k]).all()
 checked.append({'path':str(path),'input_sha256':h,'split':split})
raw=yaml.safe_load((ops/'recipe.yaml').read_text())
raw['model'].update(target_model_path='/gpfs/shared-model-weight/models/deepseek-ai/DeepSeek-V4.1-Flash-1017',draft_model_config=str(ops/'draft-config.json'))
raw['data'].update(hidden_states_path=str(inputs/'train-features'),eval_hidden_states_path=str(inputs/'validation-features'),cache_dir=str(root/'cache/data'))
raw['output_dir']=str(root/'training');raw['deployment']['trainer'].update(master_addr=master,node_rank=rank)
(node/'train.yaml').write_text(yaml.safe_dump(raw,sort_keys=False));cfg=load_config(str(node/'train.yaml'));cfg.validate_world_size(32)
assert cfg.training.optimizer=='v41_flash_muon' and cfg.training.accumulation_steps==4 and cfg.training.total_steps==118
if rank==0:(root/'config.yaml').write_text(yaml.safe_dump(cfg.model_dump(mode='json'),sort_keys=False))
(node/'input-check.json').write_text(json.dumps({'passed':True,'checked_samples':checked,'compatibility':receipt},indent=2)+'\n')
print(json.dumps({'node':rank,'inputs_passed':True,'master':master,'master_ip':json.loads((node/'rendezvous.json').read_text())['master_ip']}),flush=True)
