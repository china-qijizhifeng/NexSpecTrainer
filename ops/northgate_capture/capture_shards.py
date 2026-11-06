"""Shared-disk barriers and exact coverage for independent TP8 collectors."""
import json
from pathlib import Path
import time


def partition(samples, workers):
    """Balance complete records by token count without changing their contents."""
    assert workers > 0
    shards = [[] for _ in range(workers)]
    totals = [0] * workers
    for entry in sorted(samples, key=lambda x: (-x['tokens'], x['id'])):
        rank = min(range(workers), key=lambda i: (totals[i], i))
        shards[rank].append(entry)
        totals[rank] += entry['tokens']
    return [sorted(shard, key=lambda x: (x['tokens'], x['id'])) for shard in shards]


def check_failures(root, workers):
    for rank in range(workers):
        path = root / f'worker-{rank}' / 'REJECTED.json'
        if path.exists():
            raise RuntimeError(f'Worker {rank} rejected: {path.read_text()}')


def wait_for(root, workers, name, timeout):
    deadline = time.monotonic() + timeout
    paths = [root / f'worker-{rank}' / name for rank in range(workers)]
    while True:
        check_failures(root, workers)
        if all(path.exists() for path in paths):
            return [json.loads(path.read_text()) for path in paths]
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Waiting for {name}: {[str(p) for p in paths if not p.exists()]}')
        time.sleep(5)


def combine(root, samples, workers, manifest_sha):
    shards = partition(samples, workers)
    receipts = []
    for rank, expected in enumerate(shards):
        folder = root / f'worker-{rank}'
        result = json.loads((folder / 'RESULT.json').read_text())
        rows = json.loads((folder / 'capture-manifest.json').read_text())['records']
        assert result['passed'] and result['manifest_sha256'] == manifest_sha
        assert len(rows) == result['records'] == len(expected)
        assert [x['id'] for x in rows] == [x['id'] for x in expected]
        assert result['tokens'] == sum(x['tokens'] for x in expected)
        for row, entry in zip(rows, expected):
            assert row['manifest_sha256'] == manifest_sha
            assert row['input_tokens'] == entry['tokens'] and row['truncated'] is False
            assert row['attn_cp_size'] == 1 and row['native_roundtrip_exact']
            assert row['brotli_quality'] == 10
            for key in ['token_sha256', 'loss_mask_sha256', 'source_record_sha256']:
                assert row[key] == entry['metadata'][key]
            path = Path(row['file'])
            assert path.parent == folder / 'features' / entry['id']
            assert path.is_file() and path.stat().st_size > 0
            assert json.loads((path.parent / 'VERIFIED.json').read_text()) == row
        assert result['bytes'] == sum(Path(row['file']).stat().st_size for row in rows)
        receipts.extend(rows)
    assert len({row['id'] for row in receipts}) == len(samples) == len(receipts)
    assert {row['id'] for row in receipts} == {entry['id'] for entry in samples}
    return receipts
