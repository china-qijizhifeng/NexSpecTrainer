"""Reuse only durable trajectory receipts; capture the exact remaining partition."""
import gzip,hashlib,json,os,shutil
from pathlib import Path
import yaml
from specforge.config import Config
from specforge.application import resolve_offline_capture
from specforge.training.capture_contract import resolve_server_capture_contract
from specforge.inference.capture import CaptureConfig
from scripts.prepare_server_hidden_states import check_capacity

root=Path(os.environ['RUN_ROOT']);repo=Path(os.environ['REPO_ROOT']);migration=Path(os.environ['MIGRATION_ROOT']);old=Path(os.environ['PREVIOUS_CAPTURE'])
rank=int(os.environ['KUBEBRAIN_REPLICA']);assert int(os.environ['KUBEBRAIN_REPLICA_TOTAL'])==4
assert json.loads((old/'run.json').read_text())['yicloud']['terminal_phase'] in ('Stopped','Succeeded')
work=root/f'capture-w{rank}';work.mkdir(exist_ok=True);(work/'capture').mkdir(exist_ok=True)
manifest=json.loads((migration/'full-data/inputs-manifest.json').read_text());expected={x['id']:x for x in manifest['samples'] if x['worker']==rank}
archive=migration/f'full-data/samples-w{rank}.jsonl.gz'
for file in json.loads((migration/'full-data/transfer-manifest.json').read_text()):
 path=migration/'full-data'/file['path']
 if path==archive or path.name=='inputs-manifest.json':
  with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==file['sha256']
for d in ['sliced','features','train-features','validation-features']:(work/d).mkdir(exist_ok=True)
reused=set();remaining=[];parity=None
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location('stream_schedule',repo/'ops/yicloud-stream/schedule.py');schedule=module_from_spec(spec);spec.loader.exec_module(schedule)
# The largest reused sample provides a conservative paired full-model check.
for sample in sorted(expected.values(),key=lambda s:-s['num_tokens']):
 receipt=old/f'capture-w{rank}/sliced'/f"{sample['id']}.json"
 if not receipt.exists():continue
 data=json.loads(receipt.read_text());windows=schedule.validate_receipt(data,sample,old)
 (work/'sliced'/receipt.name).symlink_to(receipt)
 for w in windows:(work/(w['split']+'-features')/w['id']).symlink_to(w['file'])
 reused.add(sample['id'])
 if parity is None:parity=sample
seen=set();parity_row=None
with gzip.open(archive,'rt') as source,(work/'samples.jsonl').open('w') as out:
 for line in source:
  row=json.loads(line);sample=expected[row['id']];assert row['id'] not in seen;seen.add(row['id'])
  assert row['metadata']['token_sha256']==sample['token_sha256'] and len(row['input_ids'])==sample['num_tokens']
  if row['id'] not in reused:out.write(line);remaining.append(sample)
  if row['id']==parity['id']:parity_row=row
assert seen==set(expected) and parity_row is not None
(work/'parity-sample.jsonl').write_text(json.dumps(parity_row)+'\n')
assert shutil.disk_usage(root).free>21_000_000_000_000
cfg=yaml.safe_load((repo/'ops/yicloud-capture/capture.yaml').read_text())
cfg['model'].update(target_model_path='/gpfs/shared-model-weight/models/deepseek-ai/DeepSeek-V4.1-Flash-1017',draft_model_config=str(repo/'configs/deepseek-v41-flash-native.json'))
cfg['data'].update(hidden_states_path=str(work/'features'),cache_dir=str(work/'cache/data'),max_length=262144)
cfg.update(run_id='yicloud-denovo-full-capture-tuned',output_dir=str(work/'unused-training'))
(work/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
if rank==0:(root/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
parsed=Config.from_file(str(work/'config.yaml'));plan=resolve_offline_capture(parsed);contract=resolve_server_capture_contract(parsed,algorithm=plan.run.algorithm)
capture=CaptureConfig.from_strategy(plan.layout.output_names,contract.aux_layer_ids,target_repr='hidden_state',target_hidden_size=contract.target_hidden_size,target_vocab_size=contract.target_vocab_size,draft_vocab_size=contract.draft_vocab_size)
maximum=max([s['num_tokens'] for s in remaining]+[parity['num_tokens']])
# The largest trajectory is already captured, but parity replays it at depth 1.
size=check_capacity(max([s['num_tokens'] for s in remaining],default=1),capture=capture,layout=plan.layout,client_bytes=16<<30,server_bytes=16<<30,wire_encoding='bf16_fields_zstd_v2',pipeline_depth=2)
(work/'input-check.json').write_text(json.dumps(dict(passed=True,worker=rank,reused=len(reused),remaining=len(remaining),max_tokens=maximum,decoded_bytes=size,parity_id=parity['id'],parity_tokens=parity['num_tokens'],path='fused indexer + v2 compressed wire + codec8 + pipeline2; direct verified slices')))
print((work/'input-check.json').read_text(),flush=True)
