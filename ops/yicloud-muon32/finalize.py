import json,os,time
from pathlib import Path
from specforge.training.nfc_checkpoint import read_manifest
r=Path(os.environ['RUN_DIR']);deadline=time.time()+300
while not all((r/f'training/complete-rank{i}.json').exists() for i in range(32)):
 assert time.time()<deadline,'missing rank completion receipts'
 time.sleep(1)
for i in range(32):
 d=json.loads((r/f'training/complete-rank{i}.json').read_text())
 assert d['passed'] and d['optimizer_steps']==118 and d['training_micro']==472
checkpoint=r/'training/yicloud-all-anchors-muon32-step118';m=read_manifest(checkpoint)
assert m['training']['global_step']==118 and m['training']['world_size']==32
result={'passed':True,'optimizer':'v41_flash_muon','optimizer_steps':118,'world_size':32,'train_traces':1326,'validation_traces':17,'checkpoint':str(checkpoint),'checkpoint_restore_verified':False,'model_quality_improvement_verified':False}
(r/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
