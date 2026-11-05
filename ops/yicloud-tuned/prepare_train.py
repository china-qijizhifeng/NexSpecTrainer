"""Resume the exact immutable stream at the last fully published native checkpoint."""
import json,os,shutil
from pathlib import Path
import yaml
from specforge.config.schema import Config
from specforge.training.nfc_checkpoint import read_manifest,validate_layouts
from schedule import atomic,sha256
root=Path(os.environ['RUN_DIR']);old=Path(os.environ['PREVIOUS_TRAIN']);stream=Path(os.environ['STREAM_ROOT']);capture=Path(os.environ['CAPTURE_RUN']);checkpoint=Path(os.environ['RESUME_CHECKPOINT'])
assert json.loads((old/'run.json').read_text())['yicloud']['terminal_phase'] in ('Stopped','Succeeded')
assert shutil.disk_usage(root).free>850_000_000_000
m=read_manifest(checkpoint);validate_layouts(m['layouts']);state=m['training'];step=state['global_step']
assert state['epoch']==0 and state['epoch_batch']==step*16 and state['epoch_samples']==step*16
assert state['world_size']==8 and state['effective_total_steps']==796 and 0<step<796
contract=json.loads((stream/'contract.json').read_text());assert contract==state['stream_contract']
assert contract['train_slots']==101888 and contract['validation_slots']==5896
assert contract['inputs_sha256']==sha256(Path(os.environ['MIGRATION_ROOT'])/'full-data/inputs-manifest.json')
# Publisher owns an exclusive original-stream lock for the duration of this job.
if (stream/'failed.json').exists():
 shutil.copy2(stream/'failed.json',root/'previous-stream-failed.json');(stream/'failed.json').unlink()
config=yaml.safe_load((old/'config.yaml').read_text())
config['training']['resume_from']=str(checkpoint)
config['output_dir']=str(root/'training');config['data']['cache_dir']=str(root/'cache/data')
# Preserve model paths and stream contract to match native resume provenance.
(root/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False));Config.from_file(str(root/'config.yaml')).validate_world_size(8)
atomic(root/'input-check.json',dict(passed=True,resume_step=step,checkpoint=str(checkpoint),capture_run=str(capture),stream_root=str(stream),full_epoch_steps=796,remaining_steps=796-step,stream_contract=contract))
print((root/'input-check.json').read_text(),flush=True)
