"""Non-CP full-record <=64K capture; fail closed before releasing the full set.

The original tokenized IDs/masks are authoritative. No rendering, truncation,
long-input runtime helpers or CP numerics are part of this baseline.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import numpy as np
import torch

from ops.northgate_capture import retest
from specforge.data.feature_file import load_hybrid_features

ROOT = retest.ROOT
DATA = retest.DATA
LIMIT = 65536
SERVER_CAPACITY = 69632  # Page-aligned input plus scheduler reservation/headroom.
MANIFEST_SHA = "c23a84a6a1b86fd52771c8191dcf7eb4075d2de4d47cad8bf7e21bc336f3c535"
MODEL_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"


def sha_file(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy()).hexdigest()


def select(manifest):
    samples = sorted((x for x in manifest['samples'] if x['tokens'] <= LIMIT),
                     key=lambda x: (x['tokens'], x['id']))
    assert len({x['id'] for x in samples}) == len(samples)
    return samples


def validate_input(row, entry):
    assert row['id'] == entry['id']
    n = entry['tokens']
    assert 0 < n <= LIMIT
    assert len(row['input_ids']) == len(row['loss_mask']) == n
    assert set(row['loss_mask']) <= {0, 1}
    for key, expected in [('input_ids', 'token_sha256'), ('loss_mask', 'loss_mask_sha256')]:
        digest = hashlib.sha256(np.asarray(row[key], dtype=np.int64).tobytes()).hexdigest()
        assert digest == entry['metadata'][expected], (row['id'], key)
    assert entry['metadata']['model_revision'] == MODEL_REVISION


def prepare_inputs():
    manifest_path = DATA / 'tokenized/manifest.json'
    assert (DATA / 'tokenized/COMPLETE').read_text().strip() == 'verified'
    assert sha_file(manifest_path) == MANIFEST_SHA
    manifest = json.loads(manifest_path.read_text())
    samples = select(manifest)
    assert len(samples) == 1718 and sum(x['tokens'] for x in samples) == 73820977
    retest.write(ROOT / 'selection.json', {
        'manifest_sha256': MANIFEST_SHA, 'max_input_tokens': LIMIT,
        'records': len(samples), 'tokens': sum(x['tokens'] for x in samples),
        'model_revision': MODEL_REVISION, 'samples': samples})
    folder = ROOT / 'prepared-input'
    folder.mkdir(exist_ok=True)
    expected = {x['id']: x for x in samples}
    seen = set()
    for rank in range(4):
        with gzip.open(DATA / f'tokenized/worker-{rank}.jsonl.gz', 'rt') as file:
            for line in file:
                row = json.loads(line)
                if row['id'] not in expected:
                    continue
                entry = expected[row['id']]
                assert entry['worker'] == rank and row['id'] not in seen
                validate_input(row, entry)
                retest.write(folder / f"{row['id']}.json", row)
                seen.add(row['id'])
    assert seen == set(expected)
    return samples


def verify_existing(folder, entry):
    """Resume only a fully verified receipt whose file and tensors still match."""
    receipt_path = folder / 'VERIFIED.json'
    if not receipt_path.exists():
        assert not folder.exists(), f'Incomplete output requires investigation: {folder}'
        return None
    receipt = json.loads(receipt_path.read_text())
    assert receipt['id'] == entry['id'] and receipt['manifest_sha256'] == MANIFEST_SHA
    path = Path(receipt['file'])
    assert path.parent == folder and sha_file(path) == receipt['file_sha256']
    values = load_hybrid_features(path, workers=8)
    assert path.open('rb').read(8) == b'NSACTH01'
    for name, value in values.items():
        assert tensor_sha(value) == receipt['tensors'][name]['sha256']
    payload = json.loads((ROOT / 'prepared-input' / f"{entry['id']}.json").read_text())
    assert values['input_ids'].tolist() == payload['input_ids']
    assert values['loss_mask'].tolist() == payload['loss_mask']
    return receipt


def capture(store, folder, entry, repeat=0):
    existing = verify_existing(folder, entry)
    if existing is not None:
        return existing
    payload = json.loads((ROOT / 'prepared-input' / f"{entry['id']}.json").read_text())
    validate_input(payload, entry)
    folder.mkdir()
    n = entry['tokens']
    # Passing n equal to the exact row length preserves the entire trajectory.
    values = retest.capture(store, folder, payload, n, repeat)
    receipt = json.loads((folder / f'n{n}-r{repeat}.json').read_text())
    path = Path(receipt['file'])
    assert path.open('rb').read(8) == b'NSACTH01'
    assert receipt['native_roundtrip_exact'] and receipt['brotli_quality'] == 10
    receipt.update(id=entry['id'], manifest_sha256=MANIFEST_SHA,
                   file_sha256=sha_file(path), input_tokens=n, truncated=False,
                   model_revision=MODEL_REVISION, attn_cp_size=1,
                   source_record_sha256=entry['metadata']['source_record_sha256'],
                   token_sha256=entry['metadata']['token_sha256'],
                   loss_mask_sha256=entry['metadata']['loss_mask_sha256'],
                   metadata=entry['metadata'])
    del values
    retest.write(folder / 'VERIFIED.json', receipt)
    return receipt


def main():
    from mooncake.store import MooncakeDistributedStore

    torch.set_num_threads(8)
    assert os.environ.get('NORTHGATE_LONG_CAPTURE') == '0'
    assert os.environ.get('NORTHGATE_CP_DIAGNOSTICS') is None
    assert os.environ.get('NORTHGATE_RESIDENT_DEBUG') is None
    retest.state('preparing-inputs')
    samples = prepare_inputs()
    retest.write(ROOT / 'config.yaml', {
        'model': retest.MODEL, 'model_revision': MODEL_REVISION,
        'context_length': SERVER_CAPACITY, 'max_input_tokens': LIMIT,
        'manifest_sha256': MANIFEST_SHA, 'seed': 20261106,
        'tp': 8, 'ep': 8, 'cp': 1, 'records': len(samples),
        'tokens': sum(x['tokens'] for x in samples),
        'input_policy': 'complete original tokenized input; no truncation',
        'supervision_policy': 'preserve curated loss_mask exactly',
        'codec': 'native hybrid Brotli10 sign/exponent and Zstd mantissa',
        'helpers': False, 'baseline': 'non-CP; CP equivalence not accepted',
        'acceptance': 'shortest/median/longest complete records; longest exact repeat'})
    attempt = str(time.time_ns())
    master_log = (ROOT / f'mooncake-{attempt}.log').open('x')
    master = subprocess.Popen(['mooncake_master', '--enable_http_metadata_server=true',
        '--http_metadata_server_host=127.0.0.1', '--rpc_port=35551',
        '--http_metadata_server_port=35880', '--metrics_port=35900',
        '--default_kv_lease_ttl=30m'], stdout=master_log, stderr=subprocess.STDOUT,
        start_new_session=True)
    server = store = log = None
    try:
        for _ in range(30):
            assert master.poll() is None
            try:
                with socket.create_connection(('127.0.0.1', 35551), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError('Mooncake readiness')
        store = MooncakeDistributedStore()
        assert store.setup(local_hostname='127.0.0.1',
            metadata_server='http://127.0.0.1:35880/metadata', global_segment_size=0,
            local_buffer_size=8 << 30, protocol='tcp', rdma_devices='',
            master_server_addr='127.0.0.1:35551') in (0, None)
        folder = ROOT / f'noncp64k-server-{attempt}'
        folder.mkdir()
        retest.state('starting-server', context_length=SERVER_CAPACITY)
        server, log = retest.launch(False, folder, context_length=SERVER_CAPACITY)
        acceptance = ROOT / 'acceptance'
        acceptance.mkdir(exist_ok=True)
        longest = samples[-1]
        # Test capacity before collecting the complete set; never truncate it.
        retest.state('acceptance-longest', id=longest['id'], tokens=longest['tokens'])
        first = capture(store, acceptance / f"max-r0-{longest['id']}", longest)
        second = capture(store, acceptance / f"max-r1-{longest['id']}", longest, repeat=1)
        assert first['tensors'] == second['tensors'], '64K self-repeat is not bitwise exact'
        for index in [0, len(samples) // 2]:
            entry = samples[index]
            retest.state('acceptance-representative', id=entry['id'], tokens=entry['tokens'])
            capture(store, acceptance / f"representative-{entry['id']}", entry)
        retest.write(ROOT / 'ACCEPTED.json', {
            'passed': True, 'max_real_tokens': longest['tokens'],
            'max_repeat_bitwise_exact': True, 'codec_roundtrip_exact': True,
            'brotli_quality': 10, 'attn_cp_size': 1, 'records': len(samples),
            'input_manifest_sha256': MANIFEST_SHA})
        output = ROOT / 'features'
        output.mkdir(exist_ok=True)
        receipts = []
        tokens = 0
        for index, entry in enumerate(samples):
            assert server.poll() is None
            retest.state('capture', completed=index, total=len(samples),
                         completed_tokens=tokens, id=entry['id'], tokens=entry['tokens'])
            receipt = capture(store, output / entry['id'], entry)
            receipts.append(receipt)
            tokens += entry['tokens']
            with (ROOT / f'capture-manifest-{attempt}.jsonl').open('a') as file:
                file.write(json.dumps(receipt, ensure_ascii=False) + '\n')
                file.flush()
                os.fsync(file.fileno())
        assert len(receipts) == len(samples) and tokens == 73820977
        retest.write(ROOT / 'capture-manifest.json', {'records': receipts})
        retest.write(ROOT / 'RESULT.json', {
            'passed': True, 'records': len(receipts), 'tokens': tokens,
            'bytes': sum(Path(x['file']).stat().st_size for x in receipts),
            'manifest_sha256': MANIFEST_SHA, 'attn_cp_size': 1,
            'native_roundtrip_exact': True, 'brotli_quality': 10})
        retest.state('complete', passed=True, records=len(receipts), tokens=tokens)
    except BaseException as error:
        retest.write(ROOT / 'REJECTED.json', {
            'passed': False, 'error': repr(error),
            'progress': json.loads((ROOT / 'progress.json').read_text())})
        raise
    finally:
        retest.stop(server)
        if log is not None:
            log.close()
        if store is not None and hasattr(store, 'close'):
            store.close()
        retest.stop(master)
        master_log.close()


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, interrupted)
    main()
