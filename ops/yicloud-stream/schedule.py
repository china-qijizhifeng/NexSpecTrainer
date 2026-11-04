"""Durable full-epoch schedule published only from verified trajectory receipts."""
import hashlib
import json
import os
import random
import time
from pathlib import Path

QUANTUM = 128  # 8 ranks * 16 accumulation microbatches, batch size 1
RESERVE = 8192
SEED = 20261105


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_receipt(receipt, sample, capture):
    assert receipt['complete'] is True
    assert receipt['input'] == {k: v for k, v in sample.items() if k not in {'id', 'worker', 'file_sha256', 'file_bytes'}}, 'receipt metadata differs from immutable input plan'
    windows = receipt['windows']
    assert len(windows) == len(sample['slice_starts'])
    for window, start in zip(windows, sample['slice_starts']):
        name = f"{sample['id']}-s{start:06d}.ckpt.zst"
        expected = capture / f"capture-w{sample['worker']}" / (sample['split'] + '-features') / name
        assert window['id'] == name and window['split'] == sample['split']
        assert window['trajectory_id'] == sample['trajectory_id'] and window['instance_id'] == sample['instance_id']
        assert window['target_context'] == [0, sample['num_tokens']]
        assert window['training_slice'] == [start, min(start + 4096, sample['num_tokens'])]
        assert window['codec_bit_exact'] is True and window['prefix_loss_masked'] == (128 if start else 0)
        assert Path(window['file']) == expected and not expected.is_symlink()
        assert expected.stat().st_size == window['bytes']
        assert len(window['sha256']) == len(window['input_token_sha256']) == 64
    return windows


class Publisher:
    def __init__(self, capture, output, plan, reserve=RESERVE):
        self.capture, self.output, self.plan = Path(capture), Path(output), plan
        self.reserve = reserve
        self.samples = {s['id']: s for s in plan['samples']}
        self.seen, self.windows, self.pending = set(), {}, {}
        self.blocks = []
        self.published = set()
        self.started = False
        self.finished = False
        self.output.mkdir(parents=True, exist_ok=True)
        # No silent rewrite/reorder on a process restart. A resumed consumer uses
        # the same immutable blocks; producer recovery needs an explicit audit.
        assert not list((self.output / 'train').glob('*.json'))

    def scan(self):
        for rank in range(4):
            for path in sorted((self.capture / f'capture-w{rank}' / 'sliced').glob('*.json')):
                if path.stem in self.seen:
                    continue
                sample = self.samples[path.stem]
                assert sample['worker'] == rank
                windows = validate_receipt(json.loads(path.read_text()), sample, self.capture)
                for window in windows:
                    assert window['id'] not in self.windows
                    self.windows[window['id']] = window
                    if window['split'] == 'train':
                        self.pending[window['id']] = window
                self.seen.add(path.stem)

    def block(self, windows):
        assert len(windows) == QUANTUM
        index = len(self.blocks)
        assert not (self.output / 'train' / f'{index:06d}.json').exists()
        atomic(self.output / 'train' / f'{index:06d}.json', windows)
        self.blocks.append([w['id'] for w in windows])
        if index == 0:
            atomic(self.output / 'ready.json', {'first_block': 0, 'buffered_windows': len(self.pending), 'time': time.time()})

    def publish_available(self, complete=False):
        if not self.started:
            if len(self.pending) < 2 * self.reserve and not complete:
                return
            self.started = True
        while len(self.pending) >= QUANTUM + (0 if complete else self.reserve):
            # Persisted order is the replay authority; seed alone is not enough
            # because arrival timing affects membership of this reservoir.
            rng = random.Random(SEED + len(self.blocks))
            ids = rng.sample(sorted(self.pending), QUANTUM)
            self.block([self.pending.pop(key) for key in ids])
            self.published.update(ids)

    def finish(self):
        assert len(self.seen) == len(self.samples)
        train = sorted((w for w in self.windows.values() if w['split'] == 'train'), key=lambda w: w['id'])
        validation = sorted((w for w in self.windows.values() if w['split'] == 'validation'), key=lambda w: w['id'])
        assert len(train) == self.plan['train_windows']
        assert len(validation) == self.plan['validation_windows']
        self.publish_available(complete=True)
        duplicates = [dict(w, id=f'padding-{i:04d}.ckpt.zst', source=w['id']) for i, w in enumerate(train[:self.plan['padding_duplicates']])]
        tail = list(self.pending.values()) + duplicates
        random.Random(SEED + len(self.blocks)).shuffle(tail)
        assert len(tail) == QUANTUM
        self.block(tail)
        self.published.update(self.pending)
        self.pending.clear()
        assert self.published == {w['id'] for w in train}
        assert len(self.blocks) == self.plan['total_steps']
        for index in range(0, len(validation), QUANTUM):
            atomic(self.output / 'validation' / f'{index // QUANTUM:06d}.json', validation[index:index + QUANTUM])
        atomic(self.output / 'windows-manifest.json', {'windows': list(self.windows.values()), 'duplicates': duplicates})
        atomic(self.output / 'complete.json', {'passed': True, 'train_windows': len(train), 'validation_windows': len(validation), 'padding_duplicates': len(duplicates), 'total_steps': len(self.blocks), 'counts': self.plan['counts'], 'time': time.time()})
        self.finished = True

    def tick(self):
        run = json.loads((self.capture / 'run.json').read_text())
        phase = run['yicloud'].get('terminal_phase')
        assert phase not in ('Failed', 'Stopped', 'Canceled', 'Cancelled'), f'capture terminated: {phase}'
        self.scan()
        if phase == 'Succeeded':
            for rank in range(4):
                result = json.loads((self.capture / f'capture-w{rank}' / 'RESULT.json').read_text())
                assert result['passed'] and result['complete_partition'] and result['worker'] == rank
            self.finish()
        else:
            self.publish_available()
        atomic(self.output / 'status.json', {'time': time.time(), 'trajectories': len(self.seen), 'ready_train_windows': len(self.published), 'buffered_train_windows': len(self.pending), 'published_steps': len(self.blocks), 'complete': self.finished})


def main():
    root = Path(os.environ['STREAM_ROOT'])
    try:
        plan = json.loads((Path(os.environ['MIGRATION_ROOT']) / 'full-data/inputs-manifest.json').read_text())
        publisher = Publisher(os.environ['CAPTURE_RUN'], root, plan)
        while not publisher.finished:
            publisher.tick()
            if not publisher.finished:
                time.sleep(15)
    except BaseException as error:
        atomic(root / 'failed.json', {'error': repr(error), 'time': time.time()})
        raise


if __name__ == '__main__':
    main()
