"""Import completed files from a terminal run without decoding or copying tensors."""
import argparse
import json
import os
from pathlib import Path


def import_files(source, target, terminal_receipt, *, execute=False):
    old = json.loads((source / 'run.json').read_text())
    new = json.loads((target / 'run.json').read_text())
    terminal = json.loads(terminal_receipt.read_text())
    assert terminal['job_id'] == old['yicloud']['job_id']
    assert terminal['terminal'] and terminal['job_terminal'] and terminal['replicas_terminal']
    assert not new['yicloud'].get('job_id'), 'Target already submitted'
    old_tasks = old['yicloud']['payload']['TaskSpecs']
    new_tasks = new['yicloud']['payload']['TaskSpecs']
    assert old_tasks['t0']['Replicas'] == new_tasks['t0']['Replicas']
    assert old_tasks['t0']['Image'] == new_tasks['t0']['Image']
    count = total = 0
    for rank in range(new_tasks['t0']['Replicas']):
        origin = source / f'worker-{rank}'
        assert json.loads((origin / 'ACCEPTED.json').read_text())['passed']
        shard = {x['id']: x for x in json.loads((origin / 'shard.json').read_text())['samples']}
        for folder in sorted((origin / 'features').iterdir()):
            receipt = folder / 'CAPTURED.json'
            if not receipt.exists():
                receipt = folder / 'VERIFIED.json'
            if not receipt.exists():
                continue  # Preserve interrupted writes in the old run; recollect them.
            row = json.loads(receipt.read_text())
            entry = shard[row['id']]
            assert folder.name == row['id'] and row['input_tokens'] == entry['tokens']
            for key in ['token_sha256', 'loss_mask_sha256', 'source_record_sha256']:
                assert row[key] == entry['metadata'][key]
            path = Path(row['file'])
            assert path.parent == folder and path.is_file()
            size = path.stat().st_size
            assert size > 0 and size == row.get('file_bytes', size)
            destination = target / f'worker-{rank}' / 'features' / row['id']
            assert not destination.exists(), destination
            if execute:
                destination.mkdir(parents=True)
                linked = destination / path.name
                os.link(path, linked)
                row.update(file=str(linked), file_bytes=size, receipt_name='CAPTURED.json',
                           status='reused', reused_from=str(receipt),
                           readback_performed=False, prior_native_roundtrip_exact=row.get('native_roundtrip_exact'),
                           native_roundtrip_exact=None)
                temporary = destination / 'CAPTURED.partial'
                temporary.write_text(json.dumps(row, indent=2) + '\n')
                temporary.replace(destination / 'CAPTURED.json')
            count += 1
            total += entry['tokens']
    report = dict(source=str(source), target=str(target), records=count, tokens=total,
                  execute=execute, transfer='hardlink', readback_performed=False)
    if execute:
        (target / 'RESUMED.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--terminal-receipt', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(import_files(args.source, args.target, args.terminal_receipt,
                                  execute=args.execute), indent=2))
