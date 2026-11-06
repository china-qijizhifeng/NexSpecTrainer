"""Opt-in stable native partition; build once in a source-addressed local cache."""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np


@lru_cache(maxsize=1)
def _library():
    source = Path(__file__).with_suffix('.c')
    identity = hashlib.sha256(source.read_bytes() + platform.machine().encode()).hexdigest()[:24]
    root = Path(os.environ.get('SPECFORGE_NATIVE_CACHE', tempfile.gettempdir())) / 'specforge-hybrid'
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'bucket-{identity}.so'
    if not path.exists():
        fd, temporary = tempfile.mkstemp(dir=root, suffix='.so')
        os.close(fd)
        try:
            subprocess.run([os.environ.get('CC', 'cc'), '-O3', '-std=c99', '-shared',
                            '-fPIC', str(source), '-o', temporary], check=True,
                           capture_output=True, text=True)
            os.replace(temporary, path)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()
    library = ctypes.CDLL(str(path))
    function = library.hybrid_bucket
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                        ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
    function.restype = ctypes.c_int
    return library


def partition(values, contexts, buckets):
    values = np.ascontiguousarray(values, dtype=np.uint8).reshape(-1)
    contexts = np.ascontiguousarray(contexts, dtype=np.uint16).reshape(-1)
    if values.size != contexts.size or not 1 <= buckets <= 512:
        raise ValueError('Invalid hybrid partition shape or bucket count')
    output = np.empty_like(values)
    # Match np.bincount's signed dtype; mixing uint64 offsets with Python zero
    # promotes NumPy concatenation to float64 and breaks slice indices.
    counts = np.empty(buckets, dtype=np.int64)
    result = _library().hybrid_bucket(values.ctypes.data, contexts.ctypes.data,
                                     values.size, buckets, output.ctypes.data, counts.ctypes.data)
    if result:
        raise ValueError(f'Invalid hybrid partition contexts: {result}')
    return output, counts


if __name__ == '__main__':
    print(_library()._name)
