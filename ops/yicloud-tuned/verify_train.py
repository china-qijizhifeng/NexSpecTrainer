import collections,json,os
from pathlib import Path
from specforge.training.nfc_checkpoint import read_manifest,validate_layouts
root=Path(os.environ['RUN_DIR']);capture=Path(os.environ['STREAM_ROOT']);step=796
checkpoint=root/f'training/yicloud-denovo-full-stream-step{step}'
m=read_manifest(checkpoint);validate_layouts(m['layouts'])
assert m['training']['global_step']==step and m['training']['world_size']==8 and m['training']['effective_total_steps']==step
windows=json.loads((capture/'windows-manifest.json').read_text());expected=collections.Counter(w['input_token_sha256'] for w in windows['windows'] if w['split']=='train');expected.update(w['input_token_sha256'] for w in windows['duplicates']);consumed=collections.Counter();slots=[];eval_slots=[]
assert json.loads((capture/'complete.json').read_text())['passed']
for rank in range(8):
 r=json.loads((root/f'full-result-rank{rank}.json').read_text());assert r['passed'] and r['updates']==step and r['frozen_unchanged'];consumed.update(r['consumed_input_sha256']);slots.extend(r['consumed_slots']);eval_slots.extend(r['eval_slots'])
assert consumed==expected and sum(consumed.values())==101888
assert len(slots)==101888 and set(slots)=={f'yicloud-denovo-full-stream:{i:08d}' for i in range(101888)}
assert len(eval_slots)==5896 and set(eval_slots)=={f'yicloud-denovo-full-stream-eval:{i:08d}' for i in range(5896)}
result=dict(passed=True,global_step=step,world_size=8,checkpoint=str(checkpoint),checkpoint_bytes=sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file()),all_train_windows_consumed=True,train_trajectories=4188,validation_trajectories=241,test_heldout=247,full_epoch=True,final_checkpoint_restore_verified=False)
(root/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
