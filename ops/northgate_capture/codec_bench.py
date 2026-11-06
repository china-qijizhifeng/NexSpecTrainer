"""Profile native hybrid compression on one immutable, previously captured record."""
import argparse
import json
import os
from pathlib import Path
import resource
import time

import torch
from specforge.data import hybrid_feature_file as codec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-comparison', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    env = {'torch': torch.__version__, 'affinity': sorted(os.sched_getaffinity(0)),
           'cpu_count': os.cpu_count(), 'input': str(args.input), 'cpu': {}}
    for path in ['/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/cpu.stat',
                 '/sys/fs/cgroup/cpu/cpu.cfs_quota_us', '/sys/fs/cgroup/cpu/cpu.stat']:
        if Path(path).exists():
            env['cpu'][path] = Path(path).read_text()
    (args.output / 'environment.json').write_text(json.dumps(env, indent=2))
    record = codec.load_hybrid_features(args.input, workers=8)
    metrics = {}
    for name in ['_manifest', '_context_streams', '_conditional_streams',
                 '_standard_streams', '_parallel_brotli', '_encode_streams', '_write_encoded']:
        original = getattr(codec, name)
        def measured(*a, _name=name, _fn=original, **kw):
            started = time.monotonic()
            result = _fn(*a, **kw)
            metrics[_name] = metrics.get(_name, 0) + time.monotonic() - started
            return result
        setattr(codec, name, measured)
    rows = []
    configs = [(16, 256 << 10, 0)]
    configs += [(workers, chunk << 10, 2048) for workers in [8, 16, 32]
                for chunk in [64, 256, 1024]]
    configs = [(w, c, n, False) for w, c, n in configs]
    if args.native_comparison:
        configs = [(16, 256 << 10, 0, False), (16, 256 << 10, 0, True),
                   (32, 256 << 10, 0, True), (32, 64 << 10, 0, True),
                   (32, 256 << 10, 0, True)]
    baseline_sha = None
    for workers, chunk, limit, native in configs:
        os.environ['SPECFORGE_HYBRID_NATIVE_BUCKET'] = str(int(native))
        sample = {k: (v[:, :limit].contiguous() if v.ndim == 3 else v[:limit].contiguous())
                  for k, v in record.items()} if limit else record
        metrics.clear()
        path = args.output / f'w{workers}-c{chunk}-n{limit}.hybrid'
        start = time.monotonic()
        codec.save_hybrid_features(sample, path, quality=10, chunk_size=chunk,
                                   workers=workers, interleave_layers={'hidden_states': [37, 38, 39]})
        row = {'workers': workers, 'chunk_size': chunk, 'native_bucket': native,
               'tokens': sample['input_ids'].numel(), 'seconds': time.monotonic() - start,
               'stages': dict(metrics), 'bytes': path.stat().st_size,
               'maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        if args.native_comparison:
            import hashlib
            with path.open('rb') as file:
                row['file_sha256'] = hashlib.file_digest(file, 'sha256').hexdigest()
            if baseline_sha is None:
                baseline_sha = row['file_sha256']
            if chunk == 256 << 10:
                assert row['file_sha256'] == baseline_sha, 'Native output differs from baseline'
                row['baseline_bytes_identical'] = True
        rows.append(row)
        with (args.output / 'timings.jsonl').open('a') as file:
            file.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        path.unlink()
    (args.output / 'RESULT.json').write_text(json.dumps({'passed': True, 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
