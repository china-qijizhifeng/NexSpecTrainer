"""Keep training/model provenance intact and isolate evaluation outputs."""
import hashlib
import json
import os
from pathlib import Path
import yaml
from specforge.config.schema import Config
from specforge.training.nfc_checkpoint import read_manifest, validate_layouts

root = Path(os.environ['RUN_DIR'])
training = Path(os.environ['TRAINING_RUN'])
checkpoint = Path(os.environ['EVAL_CHECKPOINT'])
stream = Path(os.environ['STREAM_ROOT'])
result = json.loads((training / 'RESULT.json').read_text())
assert result['passed'] and result['global_step'] == 796 and result['all_train_windows_consumed']
assert result['checkpoint'] == str(checkpoint)
manifest = read_manifest(checkpoint)
validate_layouts(manifest['layouts'])
assert manifest['training']['global_step'] == 796 and manifest['training']['world_size'] == 8
assert manifest['training']['stream_contract'] == json.loads((stream / 'contract.json').read_text())
assert json.loads((stream / 'complete.json').read_text())['passed']
assert not (stream / 'failed.json').exists()
for variant in ('native', 'trained'):
    cfg = yaml.safe_load((training / 'config.yaml').read_text())
    cfg['training']['resume_from'] = str(checkpoint) if variant == 'trained' else None
    cfg['output_dir'] = str(root / variant / 'unused-training-output')
    cfg['data']['cache_dir'] = str(root / 'cache/data')
    path = root / f'config-{variant}.yaml'
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    Config.from_file(str(path)).validate_world_size(8)
(root / 'config.yaml').write_text((root / 'config-trained.yaml').read_text())
(root / 'input-check.json').write_text(json.dumps({
    'passed': True, 'checkpoint': str(checkpoint), 'step': 796,
    'checkpoint_manifest_sha256': hashlib.sha256((checkpoint / 'training_state.json').read_bytes()).hexdigest(),
    'validation_windows': 5896, 'validation_trajectories': 241,
    'baseline': 'original native Flash-1017 MTP', 'optimizer_updates': 0,
    'seed_policy': 'SHA256(denovo-eval-20261105:slot_id)',
}, indent=2))
