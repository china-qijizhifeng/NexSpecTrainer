"""Validate a disjoint input partition and resolve its full capture config."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import yaml
import numpy as np

from specforge.application import resolve_offline_capture
from specforge.config import Config
from specforge.inference.capture import CaptureConfig
from specforge.training.capture_contract import resolve_server_capture_contract
from scripts.prepare_server_hidden_states import check_capacity

root=Path(os.environ['NORTHGATE_ROOT']);run=Path(os.environ['RUN_DIR']);rank=int(os.environ['KUBEBRAIN_REPLICA'])
assert int(os.environ['KUBEBRAIN_REPLICA_TOTAL'])==4 and 0<=rank<4
assert (root/'tokenized/COMPLETE').read_text().strip()=='verified'
manifest=json.loads((root/'tokenized/manifest.json').read_text())
expected={x['id']:x for x in manifest['samples'] if x['worker']==rank}
assert manifest['records']==4951
seen=set();offsets=[]
with gzip.open(root/f'tokenized/worker-{rank}.jsonl.gz','rt') as source,(run/'samples-spool.jsonl').open('wb') as output:
    for line in source:
        row=json.loads(line);entry=expected[row['id']];assert row['id'] not in seen;seen.add(row['id'])
        assert len(row['input_ids'])==len(row['loss_mask'])==entry['tokens']
        assert hashlib.sha256(np.array(row['input_ids'],dtype=np.int64).tobytes()).hexdigest()==entry['metadata']['token_sha256']
        assert hashlib.sha256(np.array(row['loss_mask'],dtype=np.int64).tobytes()).hexdigest()==entry['metadata']['loss_mask_sha256']
        offsets.append((entry['tokens'],row['id'],output.tell()))
        output.write(line.encode())
assert seen==set(expected)
# One short real sample, then longest-first: exercise capacity before bulk work.
shortest=min(offsets)
ordered=[shortest]+sorted([x for x in offsets if x!=shortest],key=lambda x:(-x[0],x[1]))
with (run/'samples-spool.jsonl').open('rb') as stream,(run/'samples.jsonl').open('wb') as output:
    for _,_,offset in ordered:
        stream.seek(offset);output.write(stream.readline())
(run/'samples-spool.jsonl').unlink()
max_tokens=max(x['tokens'] for x in expected.values())
context=((max_tokens+4095)//4096)*4096+4096
cfg=yaml.safe_load(Path('examples/configs/offline/colocated/deepseek-v41-flash-native.yaml').read_text())
cfg['model'].update(target_model_path='/gpfs/shared-model-weight/models/deepseek-ai/DeepSeek-V4.1-Flash-1017',
                    draft_model_config=str(Path('configs/deepseek-v41-flash-native.json').resolve()))
cfg['data'].update(hidden_states_path=str(run/'features'),cache_dir=str(run/'cache/data'),max_length=context)
cfg.update(run_id=f'northgate-v2-worker-{rank}',output_dir=str(run/'unused-training'))
(run/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
if rank==0:Path(os.environ['RUN_ROOT'],'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
parsed=Config.from_file(str(run/'config.yaml'));plan=resolve_offline_capture(parsed)
contract=resolve_server_capture_contract(parsed,algorithm=plan.run.algorithm)
assert contract.aux_layer_ids==(37,38,39)
capture=CaptureConfig.from_strategy(plan.layout.output_names,contract.aux_layer_ids,target_repr='hidden_state',
 target_hidden_size=contract.target_hidden_size,target_vocab_size=contract.target_vocab_size,draft_vocab_size=contract.draft_vocab_size)
size=check_capacity(max_tokens,capture=capture,layout=plan.layout,client_bytes=32<<30,server_bytes=32<<30,wire_encoding='raw',pipeline_depth=1)
info={'rank':rank,'records':len(expected),'tokens':sum(x['tokens'] for x in expected.values()),'max_tokens':max_tokens,
      'context':context,'max_decoded_bytes':size,'codec':'hybrid','brotli_quality':10}
(run/'input-check.json').write_text(json.dumps(info,indent=2)+'\n')
(run/'context.txt').write_text(str(context))
print(json.dumps(info),flush=True)
