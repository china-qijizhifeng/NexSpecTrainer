"""Controlled, fail-closed real-model repeatability and CP comparison.

Two sequential servers on the same 8-GPU node. No experimental long helpers.
Every request follows HTTP -> plugin -> raw Mooncake -> native Brotli hybrid.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

import torch

from ops.northgate_capture.cp_acceptance import compare_features
from ops.northgate_capture.wait_server import wait
from specforge.data.feature_file import load_hybrid_features, save_hybrid_features


ROOT = Path(os.environ['RUN_DIR'])
DATA = Path(os.environ['NORTHGATE_ROOT'])
MODEL = '/gpfs/shared-model-weight/models/deepseek-ai/DeepSeek-V4.1-Flash-1017'
INPUT = DATA / 'runs/northgate-v41-cp8-hybrid-32h200-20261106-r3/worker-0/features/20261029190125292905266FoWFfHyL.json'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def write(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def state(stage, **fields):
    record = {'stage': stage, 'time': time.time(), **fields}
    write(ROOT / 'progress.json', record)
    print(json.dumps(record), flush=True)


def request(endpoint, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request('http://127.0.0.1:30000' + endpoint, data=data,
                                 headers={'Content-Type': 'application/json'})
    with OPENER.open(req, timeout=timeout) as response:
        payload = response.read()
    return json.loads(payload) if body is not None else payload.decode()


def stop(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def launch(cp, folder):
    if os.environ.get('NORTHGATE_CP_DIAGNOSTIC_RUN') == '1':
        os.environ['NORTHGATE_CP_DIAGNOSTICS'] = str(folder / 'probes')
    argv = ['sglang', 'serve', '--model-path', MODEL, '--trust-remote-code',
            '--language-model-only', '--random-seed', '20261106',
            '--tp-size', '8', '--ep-size', '8', '--attention-backend', 'dsv4',
            '--moe-runner-backend', 'flashinfer_mxfp4',
            '--engram-config', '{"placement":"cpu"}', '--mem-fraction-static', '0.8',
            '--context-length', '16384', '--max-total-tokens', '16384',
            '--swa-full-tokens-ratio', '1.0', '--max-prefill-tokens', '16384',
            '--max-running-requests', '1', '--skip-tokenizer-init',
            '--chunked-prefill-size', '-1', '--disable-radix-cache',
            '--speculative-algorithm', 'DSPARK', '--enable-return-aux-hidden-states',
            '--return-aux-hidden-state-layer-ids', '37', '38', '39',
            '--return-hidden-states-mode', 'full', '--no-enable-decoder-swa-bounded-replay',
            '--cuda-graph-backend-prefill', 'disabled', '--cuda-graph-backend-decode', 'disabled',
            '--host', '127.0.0.1', '--port', '30000']
    if cp:
        argv += ['--enable-prefill-cp', '--cp-strategy', 'interleave']
    write(folder / 'argv.json', argv)
    log = (folder / 'serve.log').open('x')
    process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        wait(process.pid, folder, 3600, capture=True, cp_size=8 if cp else None)
        info = json.loads((folder / 'server-info.json').read_text())
        args = info.get('server_args', info)
        expected = {'attn_cp_size': 8 if cp else 1, 'random_seed': 20261106,
                    'language_model_only': True, 'context_length': 16384,
                    'chunked_prefill_size': -1, 'return_hidden_states_mode': 'full',
                    'enable_decoder_swa_bounded_replay': False}
        for key, value in expected.items():
            assert args.get(key) == value, (key, args.get(key), value)
    except BaseException:
        stop(process)
        log.close()
        raise
    return process, log


def capture(store, folder, payload, n, repeat):
    sample = f'n{n}-r{repeat}'
    tokens = payload['input_ids'][:n]
    mask = payload['loss_mask'][:n]
    spec = {'store_id': folder.name, 'sample_id': sample, 'gen': 1, 'replace': False,
            'features': {'aux': 'hidden_states', 'last_hidden': 'target_last_hidden_states'},
            'passthrough': [{'name': key, 'data': data, 'shape': [n], 'dtype': 'int64'}
                            for key, data in [('input_ids', tokens), ('loss_mask', mask)]]}
    assert request('/flush_cache?timeout=10', timeout=30).startswith('Cache flushed.')
    started = time.monotonic()
    result = request('/generate', {'input_ids': tokens, 'sampling_params': {
        'temperature': 0, 'max_new_tokens': 0,
        'custom_params': {'spec_capture': json.dumps(spec)}}, 'stream': False})
    meta = result['meta_info']
    assert meta.get('cached_tokens', 0) == 0
    captured = meta['spec_capture']
    while isinstance(captured, list):
        assert len(captured) == 1
        captured = captured[0]
    assert 'error' not in captured, captured
    assert captured['aux_layer_ids'] == [37, 38, 39]
    values = {}
    for name, shape, dtype in [('hidden_states', [1, n, 15360], torch.bfloat16),
                               ('target_last_hidden_states', [1, n, 5120], torch.bfloat16),
                               ('input_ids', [n], torch.int64), ('loss_mask', [n], torch.int64)]:
        entry = captured['features'][name]
        assert entry['shape'] == shape and entry['dtype'] == str(dtype).split('.')[-1]
        value = torch.empty(shape, dtype=dtype)
        size = value.numel() * value.element_size()
        assert store.register_buffer(value.data_ptr(), size) in (0, None)
        key = f'{folder.name}/{sample}/g1/{name}'
        try:
            assert store.get_into(key, value.data_ptr(), size) == size
        finally:
            store.unregister_buffer(value.data_ptr())
        assert store.remove(key, force=True) in (0, None)
        assert torch.isfinite(value).all()
        values[name] = value
    assert values['input_ids'].tolist() == tokens
    assert values['loss_mask'].tolist() == mask
    path = folder / f'{sample}.ckpt.hybrid'
    save_hybrid_features(values, path, quality=10, workers=16,
                         interleave_layers={'hidden_states': [37, 38, 39]})
    restored = load_hybrid_features(path, workers=8)
    assert all(torch.equal(values[k], restored[k]) for k in values)
    receipt = {'tokens': n, 'repeat': repeat, 'file': str(path), 'brotli_quality': 10,
               'native_roundtrip_exact': True, 'cached_tokens': meta.get('cached_tokens', 0),
               'input_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
               'seconds': time.monotonic() - started,
               'tensors': {k: {'shape': list(v.shape), 'sha256': hashlib.sha256(
                   v.contiguous().view(torch.uint8).numpy()).hexdigest()} for k, v in values.items()}}
    write(folder / f'{sample}.json', receipt)
    return values


def compare(actual, reference):
    result = compare_features(actual, reference)
    for key in ['hidden_states', 'target_last_hidden_states']:
        a, b = actual[key].squeeze(0), reference[key].squeeze(0)
        relative, cosine = [], []
        for start in range(0, len(a), 128):
            x, y = a[start:start+128].double(), b[start:start+128].double()
            nx, ny = x.square().sum(1), y.square().sum(1)
            relative.extend(((x-y).square().sum(1) / ny.clamp_min(1e-300)).sqrt().tolist())
            cosine.extend(((x*y).sum(1) / (nx*ny).sqrt().clamp_min(1e-300)).tolist())
        metrics = result['tensors'][key]
        metrics.update(exact=torch.equal(a, b),
                       row_relative_l2_p50=float(torch.tensor(relative).quantile(.5)),
                       row_relative_l2_p95=float(torch.tensor(relative).quantile(.95)),
                       row_cosine_p05=float(torch.tensor(cosine).quantile(.05)))
        metrics['passed'] = (metrics['passed'] and metrics['row_relative_l2_p95'] <= .01
                             and metrics['row_cosine_p05'] >= .99995)
    result['thresholds'].update(row_p95_max_relative_l2=.01, row_p05_min_cosine=.99995)
    result['passed'] = all(x['passed'] for x in result['tensors'].values())
    return result


def main(lengths=(4096, 14226), repeats=3):
    from mooncake.store import MooncakeDistributedStore
    torch.set_num_threads(8)
    assert os.environ.get('NORTHGATE_LONG_CAPTURE') == '0'
    payload = json.loads(INPUT.read_text())['input']['payload']
    assert len(payload['input_ids']) == len(payload['loss_mask']) == 14226
    write(ROOT / 'config.yaml', {'seed': 20261106, 'context_length': 16384,
          'model': MODEL, 'input': str(INPUT), 'input_file_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(),
          'variants': ['noncp-text-native', 'cp8-text-native'], 'lengths': list(lengths),
          'repeats': repeats, 'helpers': False, 'language_model_only': True, 'codec': 'hybrid-brotli10',
          'failure_policy': 'stop at first failed comparison'})
    master_log = (ROOT / 'mooncake.log').open('x')
    master = subprocess.Popen(['mooncake_master', '--enable_http_metadata_server=true',
        '--http_metadata_server_host=127.0.0.1', '--rpc_port=35551',
        '--http_metadata_server_port=35880', '--metrics_port=35900',
        '--default_kv_lease_ttl=30m'], stdout=master_log, stderr=subprocess.STDOUT, start_new_session=True)
    store, server = None, None
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
        for cp in [False, True]:
            folder = ROOT / ('cp8-text-native' if cp else 'noncp-text-native')
            folder.mkdir()
            state('starting', variant=folder.name)
            server, log = launch(cp, folder)
            try:
                for n in lengths:
                    first = None
                    for repeat in range(repeats):
                        state('capture', variant=folder.name, tokens=n, repeat=repeat)
                        values = capture(store, folder, payload, n, repeat)
                        if first is None:
                            first = values
                        else:
                            metrics = compare(values, first)
                            write(folder / f'comparison-repeat-n{n}-r{repeat}.json', metrics)
                            assert metrics['passed'], f'{folder.name} self repeat failed at {n} r{repeat}'
                        del values
                    if cp:
                        baseline = load_hybrid_features(ROOT / 'noncp-text-native' / f'n{n}-r0.ckpt.hybrid', workers=8)
                        metrics = compare(first, baseline)
                        write(folder / f'comparison-noncp-n{n}.json', metrics)
                        del baseline
                        assert metrics['passed'], f'CP vs non-CP failed at {n}'
                    del first
            finally:
                stop(server)
                server = None
                log.close()
        write(ROOT / 'RESULT.json', {'passed': True, 'captures': 2 * len(lengths) * repeats,
              'lengths': list(lengths), 'scope': 'short real-trace comparison; not long-input capacity'})
        state('complete', passed=True)
    except BaseException as error:
        write(ROOT / 'REJECTED.json', {'passed': False, 'error': str(error),
              'progress': json.loads((ROOT / 'progress.json').read_text()) if (ROOT / 'progress.json').exists() else None})
        raise
    finally:
        stop(server)
        if store is not None and hasattr(store, 'close'):
            store.close()
        stop(master)
        master_log.close()


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, interrupted)
    main()
