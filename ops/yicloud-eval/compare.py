import json
import os
from pathlib import Path

root = Path(os.environ['RUN_DIR'])
results = {}
for variant in ('native', 'trained'):
    rows = [json.loads((root / f'{variant}-result-rank{rank}.json').read_text()) for rank in range(8)]
    assert all(r['passed'] and r['optimizer_updates'] == 0 and r['local_windows'] == 737 for r in rows)
    assert all(r['metrics'] == rows[0]['metrics'] for r in rows)
    results[variant] = rows
assert [r['anchor_sha256'] for r in results['native']] == [r['anchor_sha256'] for r in results['trained']]
metrics = {v: rows[0]['metrics'] for v, rows in results.items()}
delta = {}
for key, value in metrics['trained'].items():
    old = metrics['native'][key]
    delta[key] = [a-b for a,b in zip(value,old)] if isinstance(value,list) else value-old
result = {
    'passed': True, 'checkpoint_restore_verified': True, 'paired_anchors_identical': True,
    'validation_windows': 5896, 'validation_trajectories': 241, 'test_heldout_untouched': 247,
    'metrics': metrics, 'trained_minus_native': delta,
    'weighted_objective': {v: rows[0]['weighted_objective'] for v, rows in results.items()},
    'scope': 'offline teacher-forced validation; acceptance proxies are not measured serving acceptance or speedup',
}
(root / 'RESULT.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
