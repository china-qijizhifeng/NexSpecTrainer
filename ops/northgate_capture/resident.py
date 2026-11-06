"""Two independent 8-GPU resident layouts with run-local debug commands."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def main():
    run = Path(os.environ['NORTHGATE_RESIDENT_RUN_DIR'])
    index = int(os.environ['KUBEBRAIN_REPLICA'])
    assert int(os.environ['KUBEBRAIN_REPLICA_TOTAL']) == 2
    cp = index == 1
    root = run / f'worker-{index}'
    root.mkdir(exist_ok=True)
    os.environ['RUN_DIR'] = str(root)
    os.environ['NORTHGATE_RESIDENT_CONTROL'] = str(root / 'control')
    os.environ['NORTHGATE_RESIDENT_DEBUG'] = '1'
    from ops.northgate_capture import retest
    from mooncake.store import MooncakeDistributedStore

    control = root / 'control'
    for name in ['queue', 'results', 'modules']:
        (control / name).mkdir(parents=True, exist_ok=True)
    config = {'seed': 20261106, 'model': retest.MODEL, 'image_layouts': ['tp8-ep8', 'cp8-ep8'],
              'context_length': 16384, 'weights_resident': True, 'debug_control': 'run-local-sha256',
              'input': str(retest.INPUT), 'input_sha256': hashlib.sha256(retest.INPUT.read_bytes()).hexdigest()}
    if index == 0:
        retest.write(run / 'config.yaml', config)
    retest.write(root / 'config.json', {**config, 'cp_size': 8 if cp else 1})
    payload = json.loads(retest.INPUT.read_text())
    master_log = (root / 'mooncake.log').open('x')
    master = subprocess.Popen(['mooncake_master', '--enable_http_metadata_server=true',
        '--http_metadata_server_host=127.0.0.1', '--rpc_port=35551',
        '--http_metadata_server_port=35880', '--metrics_port=35900',
        '--default_kv_lease_ttl=30m'], stdout=master_log, stderr=subprocess.STDOUT, start_new_session=True)
    store = server = log = None
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
        assert store.setup(local_hostname='127.0.0.1', metadata_server='http://127.0.0.1:35880/metadata',
            global_segment_size=0, local_buffer_size=8 << 30, protocol='tcp', rdma_devices='',
            master_server_addr='127.0.0.1:35551') in (0, None)
        folder = root / ('cp8-text-native' if cp else 'noncp-text-native')
        folder.mkdir()
        retest.state('starting', variant=folder.name)
        server, log = retest.launch(cp, folder)
        retest.write(root / 'READY.json', {'cp_size': 8 if cp else 1, 'server_pid': server.pid,
                     'driver_pid': os.getpid(), 'time': time.time()})
        retest.state('resident', variant=folder.name)
        seen = set()
        tainted = False
        deadline = time.monotonic() + 18000
        stopping = False
        while not stopping and time.monotonic() < deadline:
            assert server.poll() is None, 'resident server exited'
            assert master.poll() is None, 'resident Mooncake exited'
            for path in sorted((control / 'queue').glob('*.json')):
                if path.name in seen:
                    continue
                seen.add(path.name)
                command = json.loads(path.read_text())
                sequence = path.stem
                result = {'sequence': sequence, 'action': command['action'], 'started': time.time()}
                try:
                    if command['action'] == 'probe':
                        tainted = True
                        spec = {**command, 'sequence': sequence}
                        retest.write(control / 'command.json', spec)
                        receipts = control / 'commands' / sequence
                        end = time.monotonic() + 120
                        while time.monotonic() < end:
                            retest.request('/server_info', timeout=120)
                            files = list(receipts.glob('rank*.json'))
                            if len(files) == 8:
                                rank_results = [json.loads(p.read_text()) for p in files]
                                assert {x['rank'] for x in rank_results} == set(range(8))
                                assert all(x['success'] for x in rank_results), rank_results
                                result['ranks'] = rank_results
                                tainted = False
                                break
                            time.sleep(.2)
                        else:
                            raise TimeoutError('resident debug all-rank receipts')
                    elif command['action'] == 'capture':
                        assert not tainted, 'prior probe did not complete on all ranks'
                        n, repeats = int(command['tokens']), int(command.get('repeats', 1))
                        assert 0 < n <= min(16384, len(payload['input_ids']))
                        assert 1 <= repeats <= 3
                        target = root / 'requests' / sequence
                        target.mkdir(parents=True)
                        for repeat in range(repeats):
                            retest.capture(store, target, payload, n, repeat)
                        result['output'] = str(target)
                    elif command['action'] == 'stop':
                        stopping = True
                    else:
                        raise ValueError('unknown resident action')
                    result['success'] = True
                except Exception as error:
                    result.update(success=False, error=str(error))
                result['finished'] = time.time()
                retest.write(control / 'results' / path.name, result)
                if stopping:
                    break
            time.sleep(.25)
        assert stopping, 'resident debug lease expired without explicit completion'
        retest.write(root / 'RESULT.json', {'success': True, 'stopped': True})
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
