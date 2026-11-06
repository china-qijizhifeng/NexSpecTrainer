"""Bounded capture/write overlap with no post-write tensor readback."""
from contextlib import closing
import json
import os
from pathlib import Path
import struct
import time

from ops.northgate_capture import retest
from specforge.data.feature_file import save_hybrid_features, validate_hybrid_options
from specforge.inference.capture_pipeline import capture_pipeline


def options():
    result = {'quality': 10,
              'chunk_size': int(os.environ.get('NORTHGATE_HYBRID_CHUNK_SIZE', 256 << 10)),
              'workers': int(os.environ.get('NORTHGATE_COMPRESSION_WORKERS', 16))}
    validate_hybrid_options(**result)
    return result


def persist(values, path, **codec):
    """Read only the small native header; never decode the newly written tensors."""
    started = time.monotonic()
    save_hybrid_features(values, path, interleave_layers={'hidden_states': [37, 38, 39]}, **codec)
    encoded = time.monotonic()
    with path.open('rb') as file:
        assert file.read(8) == b'NSACTH01'
        size = struct.unpack('<I', file.read(4))[0]
        assert 0 < size <= 1 << 20
        file.read(32)
        header = json.loads(file.read(size))
    # Reuse the SHA computed by the format writer instead of scanning tensors again.
    tensors = {x['name']: {'shape': x['shape'], 'sha256': x['sha256']}
               for x in header['tensors']}
    from ops.northgate_capture.short_capture import sha_file
    digest = sha_file(path)
    return {'tensors': tensors, 'file_sha256': digest, 'file_bytes': path.stat().st_size,
            'encode_write_seconds': encoded - started,
            'file_hash_seconds': time.monotonic() - encoded,
            'native_roundtrip_exact': None, 'readback_performed': False}


def captures(store, output, shard, before_request):
    from ops.northgate_capture import short_capture as short
    codec = options()
    # One complete V4.1 record is 2.50 GiB at 65,536 tokens; depth two fits 8 GiB.
    assert 2 * short.LIMIT * (15360 * 2 + 5120 * 2 + 16) < 8 << 30

    def produce(entry):
        before_request()
        folder = output / entry['id']
        completed = folder / 'CAPTURED.json'
        if completed.exists():
            row = json.loads(completed.read_text())
            assert row['id'] == entry['id'] and row['manifest_sha256'] == short.MANIFEST_SHA
            assert row['token_sha256'] == entry['metadata']['token_sha256']
            assert row['loss_mask_sha256'] == entry['metadata']['loss_mask_sha256']
            path = Path(row['file'])
            assert path.parent == folder and path.stat().st_size == row['file_bytes']
            return row
        assert not folder.exists(), f'Incomplete capture requires investigation: {folder}'
        folder.mkdir()
        payload = json.loads((short.ROOT / 'prepared-input' / f"{entry['id']}.json").read_text())
        short.validate_input(payload, entry)
        values, timing = retest.fetch_capture(store, folder, payload, entry['tokens'], 0,
                                              check_finite=False)
        timing['produced_at'] = time.time()
        return values, timing

    def consume(entry, produced):
        if isinstance(produced, dict):
            return produced
        values, timing = produced
        path = output / entry['id'] / f"n{entry['tokens']}-r0.ckpt.hybrid"
        consume_started = time.time()
        row = persist(values, path, **codec)
        row.update(id=entry['id'], file=str(path), input_tokens=entry['tokens'],
                   manifest_sha256=short.MANIFEST_SHA, model_revision=short.MODEL_REVISION,
                   truncated=False, attn_cp_size=1, brotli_quality=codec['quality'],
                   hybrid_chunk_size=codec['chunk_size'], compression_workers=codec['workers'],
                   native_bucket=os.environ.get('SPECFORGE_HYBRID_NATIVE_BUCKET') == '1',
                   pipeline_depth=2, receipt_name='CAPTURED.json',
                   status='captured', metadata=entry['metadata'],
                   token_sha256=entry['metadata']['token_sha256'],
                   loss_mask_sha256=entry['metadata']['loss_mask_sha256'],
                   source_record_sha256=entry['metadata']['source_record_sha256'],
                   request_seconds=timing['request_seconds'],
                   transfer_seconds=timing['transfer_seconds'],
                   queue_seconds=consume_started - timing['produced_at'],
                   produced_at=timing['produced_at'], consume_started_at=consume_started,
                   written_at=time.time(), seconds=time.monotonic() - timing['started'])
        retest.write(path.parent / 'CAPTURED.json', row)
        return row

    def discard(entry, produced):
        # fetch_capture already removed all Mooncake keys; only CPU tensors remain.
        if not isinstance(produced, dict):
            produced[0].clear()

    with closing(capture_pipeline(shard, produce=produce, consume=consume,
                                  discard=discard, depth=2)) as results:
        yield from results
