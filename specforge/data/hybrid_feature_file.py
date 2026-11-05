"""Lossless conditional BF16 activation files with Brotli and Zstd."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from specforge.data import feature_file

_MAGIC = b"NSACTH01"
_STREAM_HEADER = struct.Struct("<BI")
_FRAME_HEADER = struct.Struct("<II")
_ZSTD = 1
_BROTLI = 2
_ZSTD_CHUNK_SIZE = 1 << 20
_SIGN_CONTEXTS = 512
_EXPONENT_CONTEXTS = 256
_MANTISSA_CONTEXTS = 256
_CONDITIONAL_STREAMS = 1 + _SIGN_CONTEXTS + 1 + _EXPONENT_CONTEXTS + _MANTISSA_CONTEXTS
_MIN_CHUNK_SIZE = 4096
_MAX_CHUNK_SIZE = 64 << 20
_MAX_LAYERS = 64


def validate_hybrid_options(
    quality=10, chunk_size=256 << 10, workers=4, layer_count=None
):
    if not 1 <= quality <= 11:
        raise ValueError("Brotli quality must be between 1 and 11")
    if not _MIN_CHUNK_SIZE <= chunk_size <= _MAX_CHUNK_SIZE:
        raise ValueError("Hybrid chunk size must be between 4096 and 64 MiB")
    if workers < 1:
        raise ValueError("Hybrid compression workers must be positive")
    if layer_count is not None and not 2 <= layer_count <= _MAX_LAYERS:
        raise ValueError(f"Hybrid compression requires 2 to {_MAX_LAYERS} layers")


def _tensor_numpy(tensor):
    return (
        tensor.detach()
        .resolve_conj()
        .resolve_neg()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
    )


def _unpackbits(data, count):
    return np.unpackbits(
        np.frombuffer(data, dtype=np.uint8), bitorder="little", count=count
    )


def _validate_header(header, max_record_bytes):
    if not isinstance(header, dict) or header.get("version") != 1:
        raise ValueError("Unsupported hybrid activation version")
    entries = feature_file._validate_entries(header, max_record_bytes)
    chunk_size = header.get("chunk_size")
    if type(chunk_size) is not int:
        raise ValueError("Invalid hybrid activation chunk size")
    validate_hybrid_options(chunk_size=chunk_size)
    for entry in entries:
        codec = entry.get("codec")
        layers = entry["interleave_layers"]
        expected = (
            "conditional"
            if layers
            else ("fields" if entry["dtype"] == "bfloat16" else "raw")
        )
        if codec != expected:
            raise ValueError("Invalid hybrid tensor codec")
        if codec == "conditional":
            validate_hybrid_options(layer_count=len(layers))
        digest = entry.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Invalid hybrid tensor checksum")
    return entries


def _manifest(record, interleave_layers, chunk_size):
    if not interleave_layers:
        raise ValueError("Hybrid compression requires explicit capture layers")
    if set(interleave_layers) - set(record):
        raise ValueError("Interleave feature missing from record")
    entries = []
    for name, tensor in record.items():
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise TypeError("Activation records require dense tensors")
        if tensor.device.type != "cpu":
            raise ValueError("Stage activation tensors on CPU before compression")
        layers = list(interleave_layers[name]) if name in interleave_layers else None
        if layers is not None:
            validate_hybrid_options(layer_count=len(layers))
        entries.append(
            {
                "name": name,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "interleave_layers": layers,
                "codec": (
                    "conditional"
                    if layers
                    else ("fields" if tensor.dtype == torch.bfloat16 else "raw")
                ),
                "sha256": hashlib.sha256(_tensor_numpy(tensor).tobytes()).hexdigest(),
            }
        )
    header = {"version": 1, "chunk_size": chunk_size, "tensors": entries}
    return _validate_header(header, feature_file._MAX_RECORD_BYTES), header


def _context_streams(values, contexts, context_count, binary=False):
    streams = [[] for _ in range(context_count)]
    for layer in range(1, values.shape[0]):
        flat_contexts = contexts[layer - 1].reshape(-1)
        order = np.argsort(flat_contexts, kind="stable")
        sorted_values = values[layer].reshape(-1)[order]
        counts = np.bincount(flat_contexts, minlength=context_count)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        for context in range(context_count):
            current = sorted_values[offsets[context] : offsets[context + 1]]
            streams[context].append(
                np.packbits(current, bitorder="little") if binary else current
            )
    return [np.concatenate(selected).astype(np.uint8) for selected in streams]


def _conditional_streams(tensor, layers):
    layer_count = len(layers)
    width = tensor.shape[-1] // layer_count
    bits = (
        tensor.detach()
        .reshape(-1)
        .view(torch.uint16)
        .numpy()
        .reshape(-1, layer_count, width)
        .transpose(1, 0, 2)
        .copy()
    )
    sign = (bits >> 15).astype(np.uint8)
    exponent = ((bits >> 7) & 255).astype(np.uint8)
    mantissa = (bits & 127).astype(np.uint8)
    sign_contexts = sign[:-1].astype(np.int32) * 256 + exponent[:-1]
    exponent_streams = _context_streams(exponent, exponent[:-1], _EXPONENT_CONTEXTS)
    exponent_delta = (
        exponent[1:].astype(np.int16) - exponent[:-1].astype(np.int16)
    ).astype(np.uint8)
    mantissa_streams = _context_streams(mantissa, exponent_delta, _MANTISSA_CONTEXTS)
    mantissa_streams[0] = np.concatenate([mantissa[0].reshape(-1), mantissa_streams[0]])
    return [
        np.packbits(sign[0].reshape(-1), bitorder="little"),
        *_context_streams(sign, sign_contexts, _SIGN_CONTEXTS, binary=True),
        exponent[0].reshape(-1),
        *exponent_streams,
        *mantissa_streams,
    ]


def _standard_streams(tensor):
    raw = _tensor_numpy(tensor)
    if tensor.dtype != torch.bfloat16:
        return [raw]
    bits = raw.view("<u2")
    return [
        np.packbits((bits >> 15).astype(np.uint8), bitorder="little"),
        ((bits >> 7) & 255).astype(np.uint8),
        (bits & 127).astype(np.uint8),
    ]


def _brotli_decompress(data, expected_size, brotli):
    decompressor = brotli.Decompressor()
    raw = decompressor.process(data, expected_size + 1)
    if len(raw) > expected_size:
        raise ValueError("Brotli frame exceeds declared size")
    while not decompressor.is_finished():
        chunk = decompressor.process(b"", expected_size + 1 - len(raw))
        if not chunk:
            raise ValueError("Truncated Brotli activation frame")
        raw += chunk
        if len(raw) > expected_size:
            raise ValueError("Brotli frame exceeds declared size")
    if len(raw) != expected_size:
        raise ValueError("Invalid Brotli activation frame length")
    return raw


def _parallel_brotli(tasks, brotli, quality, workers, decode=False):
    def run(index):
        try:
            if decode:
                return index, _brotli_decompress(
                    tasks[index][1], tasks[index][0], brotli
                )
            return index, brotli.compress(
                tasks[index][1], quality=quality, lgwin=22, mode=brotli.MODE_GENERIC
            )
        except brotli.error as exc:
            raise ValueError("Corrupt Brotli activation frame") from exc

    order = sorted(range(len(tasks)), key=lambda index: tasks[index][0], reverse=True)
    results = [None] * len(tasks)
    if workers == 1 or len(tasks) < 2:
        for index in order:
            index, value = run(index)
            results[index] = value
        return results
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, value in pool.map(run, order):
            results[index] = value
    return results


def _encode_streams(streams, codecs, zstd, brotli, quality, chunk_size, workers):
    encoded = [[] for _ in streams]
    tasks = []
    for stream_index, (raw, codec) in enumerate(zip(streams, codecs)):
        raw = np.ascontiguousarray(raw).tobytes()
        if codec == _BROTLI:
            for start in range(0, len(raw), chunk_size):
                chunk = raw[start : start + chunk_size]
                tasks.append((len(chunk), stream_index, chunk))
        elif raw:
            for start in range(0, len(raw), _ZSTD_CHUNK_SIZE):
                chunk = raw[start : start + _ZSTD_CHUNK_SIZE]
                encoded[stream_index].append((len(chunk), zstd.compress(chunk)))
    brotli_tasks = [(size, chunk) for size, _, chunk in tasks]
    compressed = _parallel_brotli(brotli_tasks, brotli, quality, workers)
    for (_, stream_index, chunk), result in zip(tasks, compressed):
        encoded[stream_index].append((len(chunk), result))
    return [(codec, frames) for codec, frames in zip(codecs, encoded)]


def _write_encoded(stream, encoded):
    for codec, frames in encoded:
        stream.write(_STREAM_HEADER.pack(codec, len(frames)))
        for raw_size, compressed in frames:
            stream.write(_FRAME_HEADER.pack(raw_size, len(compressed)))
            stream.write(compressed)


def _decoded_budget(entry):
    shape = entry["shape"]
    count = math.prod(shape)
    if entry["codec"] == "raw":
        return count * feature_file._DTYPES[entry["dtype"]].itemsize
    if entry["codec"] == "fields":
        return (count + 7) // 8 + 2 * count
    layer_count = len(entry["interleave_layers"])
    leading = math.prod(shape[:-1])
    width = shape[-1] // layer_count
    count = leading * width
    packed_contexts = (count + 7 * min(count, _SIGN_CONTEXTS) + 7) // 8 if count else 0
    return (
        (count + 7) // 8 + 2 * layer_count * count + (layer_count - 1) * packed_contexts
    )


def _read_encoded(stream, stream_count, chunk_size, max_record_bytes, decoded_budget):
    encoded = []
    total_raw = total_compressed = 0
    min_chunk = min(chunk_size, _ZSTD_CHUNK_SIZE)
    max_frames = 2 * stream_count + decoded_budget // min_chunk + 1
    compressed_budget = decoded_budget + (decoded_budget >> 8) + max_frames * 1024
    for _ in range(stream_count):
        codec, frame_count = _STREAM_HEADER.unpack(
            feature_file._read_exact(stream, _STREAM_HEADER.size)
        )
        frame_size = chunk_size if codec == _BROTLI else _ZSTD_CHUNK_SIZE
        if (
            codec not in (_ZSTD, _BROTLI)
            or frame_count > decoded_budget // frame_size + 1
        ):
            raise ValueError("Invalid hybrid activation stream")
        frames = []
        for frame_index in range(frame_count):
            raw_size, compressed_size = _FRAME_HEADER.unpack(
                feature_file._read_exact(stream, _FRAME_HEADER.size)
            )
            compressed_bound = raw_size + (raw_size >> 8) + 1024
            if raw_size > max_record_bytes or compressed_size > compressed_bound:
                raise ValueError("Oversized hybrid activation frame")
            if raw_size == 0 or raw_size > frame_size:
                raise ValueError("Invalid hybrid activation chunk size")
            if frame_index + 1 < frame_count and raw_size != frame_size:
                raise ValueError("Non-final hybrid activation chunk is not full")
            if compressed_size == 0:
                raise ValueError("Invalid empty hybrid activation frame")
            total_raw += raw_size
            total_compressed += compressed_size
            if total_raw > decoded_budget or total_compressed > compressed_budget:
                raise ValueError("Hybrid tensor exceeds decoded stream budget")
            compressed = feature_file._read_exact(stream, compressed_size)
            frames.append((raw_size, compressed))
        encoded.append((codec, frames))
    return encoded


def _decode_encoded(encoded, zstd, brotli, workers):
    decoded = [[None] * len(frames) for _, frames in encoded]
    tasks = []
    for stream_index, (codec, frames) in enumerate(encoded):
        for frame_index, (raw_size, compressed) in enumerate(frames):
            if codec == _BROTLI:
                tasks.append((raw_size, stream_index, frame_index, compressed))
    compressed = _parallel_brotli(
        [(size, data) for size, _, _, data in tasks], brotli, 0, workers, decode=True
    )
    for (_, stream_index, frame_index, _), raw in zip(tasks, compressed):
        decoded[stream_index][frame_index] = raw
    decompressor = zstd.ZstdDecompressor(max_window_size=32 << 10)
    for stream_index, (codec, frames) in enumerate(encoded):
        if codec != _ZSTD:
            continue
        for frame_index, (raw_size, compressed) in enumerate(frames):
            try:
                params = zstd.get_frame_parameters(compressed)
                if (
                    params.content_size != raw_size
                    or not params.has_checksum
                    or params.window_size > 32 << 20
                ):
                    raise ValueError("Invalid hybrid Zstd frame")
                raw = decompressor.decompress(
                    compressed, max_output_size=raw_size, allow_extra_data=False
                )
            except zstd.ZstdError as exc:
                raise ValueError("Corrupt hybrid Zstd frame") from exc
            decoded[stream_index][frame_index] = raw
    streams = []
    for (codec, frames), parts in zip(encoded, decoded):
        if any(raw is None for raw in parts):
            raise ValueError("Missing hybrid activation frame")
        for raw, (expected, _) in zip(parts, frames):
            if len(raw) != expected:
                raise ValueError("Invalid hybrid activation frame length")
        streams.append(b"".join(parts))
    return streams


def _restore_context(streams, contexts, previous_offsets, binary=False, base_bytes=0):
    context_count = len(streams)
    counts = np.bincount(contexts.reshape(-1), minlength=context_count)
    sizes = (counts + 7) // 8 if binary else counts
    previous_offsets = (
        np.zeros(context_count, dtype=np.int64)
        if previous_offsets is None
        else previous_offsets
    )
    selected = []
    for context, raw in enumerate(streams):
        offset = int(previous_offsets[context]) + (base_bytes if context == 0 else 0)
        size = int(sizes[context])
        chunk = raw[offset : offset + size]
        if len(chunk) != size:
            raise ValueError("Truncated hybrid context stream")
        selected.append(
            _unpackbits(chunk, int(counts[context])).astype(np.uint8)
            if binary
            else np.frombuffer(chunk, dtype=np.uint8)
        )
    ordered = np.concatenate(selected) if selected else np.empty(0, dtype=np.uint8)
    restored = np.empty(contexts.size, dtype=np.uint8)
    order = np.argsort(contexts.reshape(-1), kind="stable")
    restored[order] = ordered
    return restored.reshape(contexts.shape), previous_offsets + sizes


def _validate_context_lengths(streams, offsets, base_bytes=0):
    for context, raw in enumerate(streams):
        if len(raw) != int(offsets[context]) + (base_bytes if context == 0 else 0):
            raise ValueError("Invalid hybrid context stream length")


def _decode_conditional(streams, entry):
    if len(streams) != _CONDITIONAL_STREAMS:
        raise ValueError("Invalid hybrid conditional stream count")
    shape = entry["shape"]
    layers = entry["interleave_layers"]
    layer_count = len(layers)
    leading = math.prod(shape[:-1])
    width = shape[-1] // layer_count
    count = leading * width
    sign = np.empty((layer_count, leading, width), dtype=np.uint8)
    exponent = np.empty_like(sign)
    mantissa = np.empty_like(sign)
    base_sign = streams[0]
    if len(base_sign) != (count + 7) // 8:
        raise ValueError("Invalid hybrid base sign stream")
    sign[0] = _unpackbits(base_sign, count).reshape(leading, width)
    sign_streams = streams[1 : 1 + _SIGN_CONTEXTS]
    exponent_base = 1 + _SIGN_CONTEXTS
    if len(streams[exponent_base]) != count:
        raise ValueError("Invalid hybrid base exponent stream")
    exponent[0] = np.frombuffer(streams[exponent_base], dtype=np.uint8).reshape(
        leading, width
    )
    exponent_streams = streams[
        exponent_base + 1 : exponent_base + 1 + _EXPONENT_CONTEXTS
    ]
    sign_offsets = exponent_offsets = None
    for layer in range(1, layer_count):
        exponent[layer], exponent_offsets = _restore_context(
            exponent_streams, exponent[layer - 1], exponent_offsets
        )
        sign_contexts = sign[layer - 1].astype(np.int32) * 256 + exponent[layer - 1]
        sign[layer], sign_offsets = _restore_context(
            sign_streams, sign_contexts, sign_offsets, binary=True
        )
    _validate_context_lengths(exponent_streams, exponent_offsets)
    _validate_context_lengths(sign_streams, sign_offsets)
    mantissa_offset = exponent_base + 1 + _EXPONENT_CONTEXTS
    mantissa_streams = streams[mantissa_offset:]
    if len(mantissa_streams) != _MANTISSA_CONTEXTS or len(mantissa_streams[0]) < count:
        raise ValueError("Invalid hybrid mantissa streams")
    mantissa[0] = np.frombuffer(mantissa_streams[0][:count], dtype=np.uint8).reshape(
        leading, width
    )
    mantissa_offsets = None
    for layer in range(1, layer_count):
        delta = (
            exponent[layer].astype(np.int16) - exponent[layer - 1].astype(np.int16)
        ).astype(np.uint8)
        mantissa[layer], mantissa_offsets = _restore_context(
            mantissa_streams, delta, mantissa_offsets, base_bytes=count
        )
    _validate_context_lengths(mantissa_streams, mantissa_offsets, base_bytes=count)
    bits = (
        (sign.astype(np.uint16) << 15)
        | (exponent.astype(np.uint16) << 7)
        | mantissa.astype(np.uint16)
    )
    tensor = torch.empty(entry["shape"], dtype=torch.bfloat16, device="cpu")
    tensor.view(torch.uint16).numpy().reshape(-1, layer_count, width)[:] = (
        bits.transpose(1, 0, 2)
    )
    return tensor


def _decode_standard(streams, entry):
    dtype = feature_file._DTYPES[entry["dtype"]]
    shape = entry["shape"]
    count = math.prod(shape)
    tensor = torch.empty(shape, dtype=dtype, device="cpu")
    raw = tensor.reshape(-1).view(torch.uint8).numpy()
    if dtype != torch.bfloat16:
        if len(streams) != 1 or len(streams[0]) != raw.size:
            raise ValueError("Invalid hybrid raw tensor stream")
        raw[:] = np.frombuffer(streams[0], dtype=np.uint8)
        return tensor
    if (
        len(streams) != 3
        or len(streams[0]) != (count + 7) // 8
        or len(streams[1]) != count
        or len(streams[2]) != count
    ):
        raise ValueError("Invalid hybrid BF16 field streams")
    sign = _unpackbits(streams[0], count).astype(np.uint16)
    exponent = np.frombuffer(streams[1], dtype=np.uint8).astype(np.uint16)
    mantissa = np.frombuffer(streams[2], dtype=np.uint8).astype(np.uint16)
    if np.any(mantissa > 127):
        raise ValueError("Invalid hybrid BF16 mantissa")
    tensor.view(torch.uint16).numpy().reshape(-1)[:] = (
        (sign << 15) | (exponent << 7) | mantissa
    )
    return tensor


def _check_tensor(entry, tensor):
    digest = hashlib.sha256(_tensor_numpy(tensor).tobytes()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"Hybrid tensor checksum mismatch: {entry['name']}")


def save_hybrid_features(
    record: Mapping[str, torch.Tensor],
    path,
    *,
    quality=10,
    chunk_size=256 << 10,
    workers=4,
    interleave_layers: Mapping[str, Sequence[int]] | None = None,
):
    zstd = feature_file.require_zstandard()
    brotli = feature_file.require_brotli()
    validate_hybrid_options(quality, chunk_size, workers)
    entries, header = _manifest(record, interleave_layers or {}, chunk_size)
    encoded_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(encoded_header) > feature_file._MAX_HEADER:
        raise ValueError("Activation manifest is too large")
    compressor = zstd.ZstdCompressor(level=1, write_checksum=True, threads=0)

    def write(stream):
        stream.write(
            _MAGIC
            + struct.pack("<I", len(encoded_header))
            + hashlib.sha256(encoded_header).digest()
            + encoded_header
        )
        for entry in entries:
            tensor = record[entry["name"]].detach().resolve_conj().resolve_neg()
            if entry["codec"] == "conditional":
                streams = _conditional_streams(tensor, entry["interleave_layers"])
                codecs = [_BROTLI] * (1 + _SIGN_CONTEXTS + 1 + _EXPONENT_CONTEXTS) + [
                    _ZSTD
                ] * _MANTISSA_CONTEXTS
            else:
                streams = _standard_streams(tensor)
                codecs = [_ZSTD] * len(streams)
            encoded = _encode_streams(
                streams, codecs, compressor, brotli, quality, chunk_size, workers
            )
            _write_encoded(stream, encoded)

    if hasattr(path, "write"):
        write(path)
        return
    path = os.fspath(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp", delete=False
        ) as stream:
            temporary = stream.name
            write(stream)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)


def load_hybrid_features(
    path, *, max_record_bytes=feature_file._MAX_RECORD_BYTES, workers=4
):
    zstd = feature_file.require_zstandard()
    brotli = feature_file.require_brotli()
    if workers < 1:
        raise ValueError("Hybrid compression workers must be positive")

    def decode(stream):
        if feature_file._read_exact(stream, len(_MAGIC)) != _MAGIC:
            raise ValueError("Invalid hybrid activation magic")
        size = struct.unpack("<I", feature_file._read_exact(stream, 4))[0]
        if size > feature_file._MAX_HEADER:
            raise ValueError("Activation manifest is too large")
        checksum = feature_file._read_exact(stream, 32)
        encoded_header = feature_file._read_exact(stream, size)
        if hashlib.sha256(encoded_header).digest() != checksum:
            raise ValueError("Corrupt activation manifest checksum")
        header = json.loads(encoded_header)
        entries = _validate_header(header, max_record_bytes)
        record = {}
        for entry in entries:
            stream_count = (
                _CONDITIONAL_STREAMS
                if entry["codec"] == "conditional"
                else (3 if entry["codec"] == "fields" else 1)
            )
            encoded = _read_encoded(
                stream,
                stream_count,
                header["chunk_size"],
                max_record_bytes,
                _decoded_budget(entry),
            )
            streams = _decode_encoded(encoded, zstd, brotli, workers)
            tensor = (
                _decode_conditional(streams, entry)
                if entry["codec"] == "conditional"
                else _decode_standard(streams, entry)
            )
            _check_tensor(entry, tensor)
            record[entry["name"]] = tensor
        if stream.read(1):
            raise ValueError("Trailing data in hybrid activation file")
        return record

    if hasattr(path, "read"):
        return decode(path)
    with open(path, "rb") as stream:
        return decode(stream)
