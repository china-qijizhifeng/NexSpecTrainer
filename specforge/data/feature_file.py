"""Offline tensor files with Zstd and conditional Brotli/Zstd compression.

The versioned Zstd format contains a bounded JSON header followed by
length-prefixed frames. V1 includes checksums; V2 explicitly omits hashing. BF16 blocks store packed sign, exponent and
mantissa separately; other dtypes retain their bytes. The hybrid format and its
parallel conditional codec live in hybrid_feature_file.py. Neither uses pickle.
"""

import gzip
import hashlib
import io
import json
import math
import os
import struct
import sys
import tempfile
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

FEATURE_SUFFIXES = (".ckpt", ".ckpt.gz", ".ckpt.zst", ".ckpt.hybrid")
_MAGIC = b"NSACTZ01"
_NO_HASH_MAGIC = b"NSACTZ02"
_BLOCK_ELEMENTS = 1 << 20
_MAX_HEADER = 1 << 20
_MAX_RECORD_BYTES = 8 << 30
_DTYPES = {
    name: getattr(torch, name)
    for name in (
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "int8",
        "uint8",
        "int16",
        "int32",
        "int64",
        "bool",
        "complex64",
        "complex128",
    )
}


def _zstd_workers(workers):
    if workers is None:
        workers = int(os.environ.get("SPECFORGE_ZSTD_WORKERS", "1"))
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("Zstd workers must be an integer between 1 and 32")
    return workers


def _ordered_blocks(function, blocks, workers):
    """Preserve frame order with at most two blocks per worker in flight."""
    if workers == 1:
        yield from map(function, blocks)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for block in blocks:
            pending.append(pool.submit(function, block))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def require_zstandard():
    """Fail early on capture startup; raw/gzip readers need no extra package."""
    try:
        import zstandard
    except ImportError as exc:
        raise ImportError(
            "Zstd activation files require: pip install 'specforge[compression]'"
        ) from exc
    if sys.byteorder != "little":
        raise RuntimeError("Zstd activation files currently require little-endian CPUs")
    return zstandard


def require_brotli():
    try:
        import brotli
    except ImportError as exc:
        raise ImportError(
            "Hybrid activation files require Brotli >= 1.2: pip install 'specforge[compression]'"
        ) from exc
    try:
        version = tuple(int(part) for part in brotli.__version__.split(".")[:2])
    except (AttributeError, ValueError) as exc:
        raise ImportError("Unable to determine Brotli version") from exc
    if version < (1, 2):
        raise ImportError("Hybrid activation files require Brotli >= 1.2")
    return brotli


def require_hybrid_compression():
    return require_zstandard(), require_brotli()


def _read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("Truncated Zstd activation file")
    return data


def _validate_entries(header, max_record_bytes):
    if (
        not isinstance(header, dict)
        or type(header.get("version")) is not int
        or header["version"] not in (1, 2)
    ):
        raise ValueError("Unsupported activation format version")
    entries = header.get("tensors")
    if not isinstance(entries, list):
        raise ValueError("Invalid tensor manifest")
    names, total = set(), 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid tensor entry")
        name, dtype, shape = (entry.get(key) for key in ("name", "dtype", "shape"))
        if not isinstance(name, str) or name in names:
            raise ValueError("Invalid or duplicate tensor name")
        names.add(name)
        if not isinstance(dtype, str) or dtype not in _DTYPES:
            raise ValueError("Unsupported tensor dtype")
        if (
            not isinstance(shape, list)
            or len(shape) > 32
            or any(type(n) is not int or n < 0 or n > max_record_bytes for n in shape)
        ):
            raise ValueError("Invalid tensor shape")
        total += math.prod(shape) * _DTYPES[dtype].itemsize
        if total > max_record_bytes:
            raise ValueError("Activation record exceeds decoded byte limit")
        if "interleave_layers" not in entry:
            raise ValueError("Missing interleave layout")
        layers = entry["interleave_layers"]
        if layers is not None and (
            not isinstance(layers, list)
            or not layers
            or any(type(n) is not int or n < 0 for n in layers)
            or len(set(layers)) != len(layers)
            or not shape
            or shape[-1] % len(layers)
            or dtype != "bfloat16"
        ):
            raise ValueError("Invalid BF16 interleave layer layout")
    return entries


def save_zstd_features(
    record: Mapping[str, torch.Tensor],
    path: str,
    *,
    level: int = 1,
    interleave_layers: Mapping[str, Sequence[int]] | None = None,
    workers: int | None = None,
    checksums: bool = True,
):
    """Save CPU tensors to an atomic file path or an already-open binary stream.

    Restore values/dtype/shape, not strides. Callers own stream publication.
    Workers compress independent blocks without changing the file bytes.
    checksums=False selects the distinct V2 format without SHA256 or Zstd checksums.
    None reads SPECFORGE_ZSTD_WORKERS (default 1); in-flight work is bounded.

    Interleave is opt-in per BF16 feature. Its last axis must be the ordered
    concatenation of equally sized layers named in ``interleave_layers``.
    """
    require_zstandard()
    workers = _zstd_workers(workers)
    if not 1 <= level <= 19:
        raise ValueError("Zstd level must be between 1 and 19")
    interleave_layers = interleave_layers or {}
    if set(interleave_layers) - set(record):
        raise ValueError("Interleave feature missing from record")
    entries = []
    for name, tensor in record.items():
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise TypeError("Activation records require dense tensors")
        if tensor.device.type != "cpu":
            raise ValueError("Stage activation tensors on CPU before compression")
        entries.append(
            {
                "name": name,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "interleave_layers": (
                    list(interleave_layers[name]) if name in interleave_layers else None
                ),
            }
        )
    header = {"version": 1 if checksums else 2, "tensors": entries}
    _validate_entries(header, _MAX_RECORD_BYTES)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_HEADER:
        raise ValueError("Activation manifest is too large")
    if hasattr(path, "write"):
        _write_zstd_stream(path, encoded, entries, record, level, workers, checksums)
    else:
        _save_zstd_path(path, encoded, entries, record, level, workers, checksums)


def _write_zstd_stream(stream, encoded, entries, record, level, workers, checksums):
    local = threading.local()

    def compress_block(item):
        block, bf16 = item
        if not hasattr(local, "compressor"):
            local.compressor = require_zstandard().ZstdCompressor(
                level=level, write_checksum=checksums, threads=0
            )
        if bf16:
            bits = block.view("<u2")
            fields = (
                np.packbits((bits >> 15).astype(np.uint8), bitorder="little"),
                ((bits >> 7) & 255).astype(np.uint8),
                (bits & 127).astype(np.uint8),
            )
        else:
            fields = (block,)
        frames = []
        for field in fields:
            compressed = local.compressor.compress(field)
            frames.extend((struct.pack("<I", len(compressed)), compressed))
        return frames

    stream.write(
        (_MAGIC if checksums else _NO_HASH_MAGIC)
        + struct.pack("<I", len(encoded))
        + (hashlib.sha256(encoded).digest() if checksums else b"")
        + encoded
    )
    for entry in entries:
        tensor = record[entry["name"]].detach().resolve_conj().resolve_neg()
        layers = entry["interleave_layers"]
        if layers:
            tensor = tensor.reshape(
                *tensor.shape[:-1], len(layers), tensor.shape[-1] // len(layers)
            ).transpose(-1, -2)
        raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy()
        itemsize = tensor.element_size()
        blocks = (
            (
                raw[start * itemsize : (start + _BLOCK_ELEMENTS) * itemsize],
                tensor.dtype == torch.bfloat16,
            )
            for start in range(0, tensor.numel(), _BLOCK_ELEMENTS)
        )
        for frames in _ordered_blocks(compress_block, blocks, workers):
            for frame in frames:
                stream.write(frame)


def _save_zstd_path(path, encoded, entries, record, level, workers, checksums):
    path = os.fspath(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp", delete=False
        ) as stream:
            temporary = stream.name
            _write_zstd_stream(
                stream, encoded, entries, record, level, workers, checksums
            )
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)


def load_zstd_features(
    path: str,
    *,
    max_record_bytes: int = _MAX_RECORD_BYTES,
    workers: int | None = None,
    expected_version: int | None = None,
):
    """Decode one record, rejecting truncation, corrupt frames and bad metadata.

    The decoded byte limit covers tensor storage, not temporary work buffers.
    Accept a file path or an already-open binary stream (owned by the caller).
    Compressed records are materialized in RAM; they cannot be memory mapped.
    """
    zstd = require_zstandard()
    workers = _zstd_workers(workers)
    local = threading.local()
    checksums = True

    def read_compressed(stream, expected):
        size = struct.unpack("<I", _read_exact(stream, 4))[0]
        if size > expected + (128 << 10):
            raise ValueError("Oversized activation frame")
        return _read_exact(stream, size), expected

    def decode_frame(frame):
        compressed, expected = frame
        if not hasattr(local, "decompressor"):
            local.decompressor = zstd.ZstdDecompressor(max_window_size=32 << 10)
        try:
            params = zstd.get_frame_parameters(compressed)
            if (
                params.content_size != expected
                or params.has_checksum != checksums
                or params.window_size > 32 << 20
            ):
                raise ValueError("Invalid activation frame size or missing checksum")
            decoded = local.decompressor.decompress(
                compressed, max_output_size=expected, allow_extra_data=False
            )
        except zstd.ZstdError as exc:
            raise ValueError("Corrupt Zstd activation frame") from exc
        if len(decoded) != expected:
            raise ValueError("Invalid decoded activation frame length")
        return np.frombuffer(decoded, dtype=np.uint8)

    def decode_block(item):
        destination, n, bf16, frames = item
        if bf16:
            sign, exponent, mantissa = map(decode_frame, frames)
            if np.any(mantissa > 127):
                raise ValueError("Invalid BF16 mantissa")
            bits = (
                np.unpackbits(sign, bitorder="little", count=n).astype(np.uint16) << 15
                | exponent.astype(np.uint16) << 7
                | mantissa.astype(np.uint16)
            )
            destination[:] = bits.view(np.uint8)
        else:
            destination[:] = decode_frame(frames[0])

    def blocks(stream, raw, count, itemsize, bf16):
        for start in range(0, count, _BLOCK_ELEMENTS):
            n = min(_BLOCK_ELEMENTS, count - start)
            sizes = ((n + 7) // 8, n, n) if bf16 else (n * itemsize,)
            frames = [read_compressed(stream, size) for size in sizes]
            yield raw[start * itemsize : (start + n) * itemsize], n, bf16, frames

    def decode_stream(stream):
        nonlocal checksums
        magic = _read_exact(stream, len(_MAGIC))
        if magic not in (_MAGIC, _NO_HASH_MAGIC):
            raise ValueError("Invalid Zstd activation magic")
        checksums = magic == _MAGIC
        version = 1 if checksums else 2
        if expected_version is not None and version != expected_version:
            raise ValueError("Activation format does not match wire encoding")
        size = struct.unpack("<I", _read_exact(stream, 4))[0]
        if size > _MAX_HEADER:
            raise ValueError("Activation manifest is too large")
        checksum = _read_exact(stream, 32) if checksums else None
        encoded = _read_exact(stream, size)
        if checksums and hashlib.sha256(encoded).digest() != checksum:
            raise ValueError("Corrupt activation manifest checksum")
        header = json.loads(encoded)
        entries = _validate_entries(header, max_record_bytes)
        if header["version"] != version:
            raise ValueError("Activation header version does not match magic")
        record = {}
        for entry in entries:
            dtype = _DTYPES[entry["dtype"]]
            count = math.prod(entry["shape"])
            tensor = torch.empty(count, dtype=dtype, device="cpu")
            raw = tensor.view(torch.uint8).numpy()
            itemsize = tensor.element_size()
            for _ in _ordered_blocks(
                decode_block,
                blocks(stream, raw, count, itemsize, dtype == torch.bfloat16),
                workers,
            ):
                pass
            shape, layers = entry["shape"], entry["interleave_layers"]
            if layers:
                tensor = (
                    tensor.reshape(*shape[:-1], shape[-1] // len(layers), len(layers))
                    .transpose(-1, -2)
                    .contiguous()
                )
            record[entry["name"]] = tensor.reshape(shape)
        if stream.read(1):
            raise ValueError("Trailing data in Zstd activation file")
        return record

    if hasattr(path, "read"):
        return decode_stream(path)
    with open(path, "rb") as stream:
        return decode_stream(stream)


def validate_hybrid_options(*args, **kwargs):
    from specforge.data.hybrid_feature_file import validate_hybrid_options as validate

    return validate(*args, **kwargs)


def save_hybrid_features(*args, **kwargs):
    from specforge.data.hybrid_feature_file import save_hybrid_features as save

    return save(*args, **kwargs)


def load_hybrid_features(*args, **kwargs):
    from specforge.data.hybrid_feature_file import load_hybrid_features as load

    return load(*args, **kwargs)


def load_feature_file(path: str) -> dict[str, torch.Tensor]:
    """Shared reader for raw/gzip checkpoints and versioned compressed records."""
    path = os.fspath(path)
    if path.endswith(".ckpt.hybrid"):
        return load_hybrid_features(path)
    if path.endswith(".ckpt.zst"):
        return load_zstd_features(path)
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as stream:
            return torch.load(io.BytesIO(stream.read()), weights_only=False)
    return torch.load(path, weights_only=False, mmap=True)
