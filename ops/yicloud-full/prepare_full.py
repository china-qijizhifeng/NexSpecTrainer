"""Validate and select one disjoint complete-trajectory capture partition."""
import gzip,hashlib,json,os,shutil
from pathlib import Path
import yaml
from specforge.config import Config
from specforge.application import resolve_offline_capture
from specforge.training.capture_contract import resolve_server_capture_contract
from specforge.inference.capture import CaptureConfig
from scripts.prepare_server_hidden_states import check_capacity
root=Path(os.environ['RUN_ROOT']);repo=Path(os.environ['REPO_ROOT']);migration=Path(os.environ['MIGRATION_ROOT'])
rank=int(os.environ['KUBEBRAIN_REPLICA']);assert int(os.environ['KUBEBRAIN_REPLICA_TOTAL'])==4
work=root/f'capture-w{rank}';work.mkdir(exist_ok=True);(work/'capture').mkdir(exist_ok=True)
manifest=json.loads((migration/'full-data/inputs-manifest.json').read_text())
assert manifest['counts']=={'train':4188,'validation':241,'test_heldout':247}
assert manifest['max_tokens']==234048
expected={x['id']:x for x in manifest['samples'] if x['worker']==rank}
archive=migration/f'full-data/samples-w{rank}.jsonl.gz'
transfers=json.loads((migration/'full-data/transfer-manifest.json').read_text())
for file in transfers:
 path=migration/'full-data'/file['path']
 if path==archive or path.name=='inputs-manifest.json':
  with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==file['sha256']
seen=set()
with gzip.open(archive,'rt') as source,(work/'samples.jsonl').open('w') as output:
 for line in source:
  row=json.loads(line);entry=expected[row['id']];assert row['id'] not in seen;seen.add(row['id'])
  assert row['metadata']['token_sha256']==entry['token_sha256'] and len(row['input_ids'])==entry['num_tokens']
  output.write(line)
assert seen==set(expected)
assert shutil.disk_usage(root).free>21_000_000_000_000
cfg=yaml.safe_load((repo/'ops/yicloud-capture/capture.yaml').read_text())
cfg['model'].update(target_model_path='/gpfs/shared-model-weight/models/deepseek-ai/DeepSeek-V4.1-Flash-1017',draft_model_config=str(repo/'configs/deepseek-v41-flash-native.json'))
cfg['data'].update(hidden_states_path=str(work/'features'),cache_dir=str(work/'cache/data'),max_length=262144)
cfg.update(run_id='yicloud-denovo-full-capture',output_dir=str(work/'unused-training'))
(work/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
if rank==0:(root/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
parsed=Config.from_file(str(work/'config.yaml'));plan=resolve_offline_capture(parsed);contract=resolve_server_capture_contract(parsed,algorithm=plan.run.algorithm)
capture=CaptureConfig.from_strategy(plan.layout.output_names,contract.aux_layer_ids,target_repr='hidden_state',target_hidden_size=contract.target_hidden_size,target_vocab_size=contract.target_vocab_size,draft_vocab_size=contract.draft_vocab_size)
assert contract.aux_layer_ids==(37,38,39)
maximum=max(x['num_tokens'] for x in expected.values())
size=check_capacity(maximum,capture=capture,layout=plan.layout,client_bytes=16<<30,server_bytes=16<<30)
(work/'input-check.json').write_text(json.dumps(dict(passed=True,worker=rank,records=len(expected),max_tokens=maximum,decoded_bytes=size)))
