import json,os,sys
from pathlib import Path
from specforge.training.nfc_checkpoint import read_manifest,validate_layouts
root=Path(os.environ['RUN_DIR']);initial=sys.argv[1]=='initial';step=1 if initial else 3
checkpoint=root/f'training/yicloud-denovo-pilot-step{step}'
m=read_manifest(checkpoint);validate_layouts(m['layouts'])
assert m['training']['global_step']==step and m['training']['world_size']==8 and m['training']['effective_total_steps']==3
phase='initial' if initial else 'resume'
for rank in range(8):
 r=json.loads((root/f'{phase}-result-rank{rank}.json').read_text());assert r['passed'] and r['updates']==(1 if initial else 2)
result={'passed':True,'global_step':step,'world_size':8,'checkpoint':str(checkpoint),'checkpoint_bytes':sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file()),'resumed_and_updated':not initial,'scope':'3 optimizer updates on 8 real 4K windows; not full DeNovoSWE training or YiCloud capture acceptance'}
(root/('first-checkpoint.json' if initial else 'RESULT.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
