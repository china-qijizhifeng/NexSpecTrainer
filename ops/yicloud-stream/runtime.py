"""Run-scoped adapter: fixed slot refs, immutable shuffled blocks, native tensors."""
import dataclasses
import json
import os
import time
from pathlib import Path

from schedule import QUANTUM, sha256


def read_slot(root, split, index, timeout=72 * 3600):
    root = Path(root)
    block = root / split / f'{index // QUANTUM:06d}.json'
    deadline = time.monotonic() + timeout
    while True:
        if (root / 'failed.json').exists():
            raise RuntimeError((root / 'failed.json').read_text())
        if block.exists():
            return json.loads(block.read_text())[index % QUANTUM]
        if (root / 'complete.json').exists():
            raise RuntimeError(f'completed stream is missing slot {split}/{index}')
        if time.monotonic() > deadline:
            raise TimeoutError(f'timed out waiting for {split}/{index}')
        time.sleep(1)


def install(root):
    import specforge.launch as launch
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader
    from specforge.runtime.data_plane.feature_store import LocalFeatureStore
    root = Path(root)
    contract = json.loads((root / 'contract.json').read_text())
    original_iter = OfflineManifestReader.__iter__
    original_shard = launch._shard_offline_refs
    original_get = LocalFeatureStore.get
    original_assemble = launch._assemble_trainer

    def iterate(self):
        path = Path(self.hidden_states_path)
        if path.parent != root or path.name not in ('train', 'validation'):
            yield from original_iter(self)
            return
        assert not self.validate_files
        for index in range(contract[path.name + '_slots']):
            ref = self._ref_for(index, str(path / f'{index:08d}.ckpt.zst'))
            ref.metadata['stream_slot'] = {'split': path.name, 'index': index}
            yield ref

    def shard(refs, **kwargs):
        if refs and 'stream_slot' in refs[0].metadata:
            assert all('stream_slot' in r.metadata for r in refs)
            assert kwargs.get('epoch', 0) == 0, 'this stream is one complete epoch'
            kwargs['shuffle'] = False  # publisher shuffled the actual windows
        return original_shard(refs, **kwargs)

    def get(self, ref, **kwargs):
        slot = ref.metadata.get('stream_slot')
        if slot is None:
            return original_get(self, ref, **kwargs)
        window = read_slot(root, slot['split'], slot['index'])
        path = Path(window['file'])
        assert path.stat().st_size == window['bytes'] and sha256(path) == window['sha256'], f'corrupt feature: {path}'
        resolved = dataclasses.replace(ref, feature_store_uri='file://' + str(path))
        return original_get(self, resolved, **kwargs)

    def assemble(**kwargs):
        refs = kwargs.get('ref_source', {}).get('refs', [])
        if refs and 'stream_slot' in refs[0].metadata:
            kwargs['checkpoint_extra'] = dict(kwargs.get('checkpoint_extra') or {}, stream_contract=contract)
        return original_assemble(**kwargs)

    OfflineManifestReader.__iter__ = iterate
    launch._shard_offline_refs = shard
    LocalFeatureStore.get = get
    launch._assemble_trainer = assemble
