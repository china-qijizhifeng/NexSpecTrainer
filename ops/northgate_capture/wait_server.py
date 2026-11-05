"""Bound cold-start waiting by wall time and observe the owned process state."""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def validate_cp(info, size):
    args = info.get('server_args', info)
    required = {'enable_prefill_cp': True, 'attn_cp_size': size,
                'cp_strategy': 'interleave', 'language_model_only': True,
                'chunked_prefill_size': -1, 'return_hidden_states_mode': 'full',
                'enable_decoder_swa_bounded_replay': False}
    for name, expected in required.items():
        if args.get(name) != expected:
            raise RuntimeError(f'CP capture contract {name}: {args.get(name)!r} != {expected!r}')


def wait(pid, output, timeout, *, capture=False, speculative=False, cp_size=None):
    output.mkdir(exist_ok=True, parents=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    started = time.time()
    while time.monotonic() < deadline:
        try:
            status = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0]
        except FileNotFoundError:
            raise RuntimeError('owned inference process exited before readiness') from None
        if status == 'Z':
            raise RuntimeError('owned inference process is a zombie before readiness')
        try:
            with opener.open('http://127.0.0.1:30000/server_info', timeout=4) as response:
                info = json.load(response)
        except (OSError, ValueError):
            (output/'readiness-progress.json').write_text(json.dumps({
                'pid': pid, 'waited_seconds': time.time()-started,
                'timeout_seconds': timeout, 'ready': False})+'\n')
            time.sleep(5)
            continue
        if capture and not any(s.get('spec_capture_backend') == 'plugin'
                               for s in info.get('internal_states', [])):
            raise RuntimeError('ready server does not advertise capture plugin')
        if cp_size is not None:
            validate_cp(info, cp_size)
        if speculative and info.get('speculative_algorithm') != 'DSPARK':
            # SGLang versions nest the algorithm under server_args.
            def algorithms(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key == 'speculative_algorithm':
                            yield item
                        yield from algorithms(item)
                elif isinstance(value, list):
                    for item in value:
                        yield from algorithms(item)
            if 'DSPARK' not in algorithms(info):
                raise RuntimeError('ready server does not advertise DSPARK')
        (output/'server-info.json').write_text(json.dumps(info, indent=2)+'\n')
        (output/'readiness-progress.json').write_text(json.dumps({
            'pid': pid, 'waited_seconds': time.time()-started,
            'timeout_seconds': timeout, 'ready': True})+'\n')
        return
    raise TimeoutError(f'owned inference readiness exceeded {timeout} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=5400)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--speculative', action='store_true')
    parser.add_argument('--cp-size', type=int)
    args = parser.parse_args()
    wait(args.pid, args.output, args.timeout, capture=args.capture,
         speculative=args.speculative, cp_size=args.cp_size)
