"""Versioned lossless Mooncake object encoding, separate from tensor FeatureSpec.

Each compressed object is a one-tensor PR #2 activation record in memory. The
metadata describes wire bytes; callers keep original shape/dtype FeatureSpecs.
No temporary file or model/training dependency is used by this module.
"""

from __future__ import annotations

import hashlib
import io
import math
import re

import torch

from specforge.data.feature_file import load_zstd_features, save_zstd_features

WIRE_ENCODING = "bf16_fields_zstd_v1"
NO_HASH_WIRE_ENCODING = "bf16_fields_zstd_v2"
WIRE_ENCODINGS = (WIRE_ENCODING, NO_HASH_WIRE_ENCODING)
WIRE_METADATA_KEY = "wire_features"
DEFAULT_MAX_DECODED_BYTES = 8 << 30


class _BufferReader:
    """Read bounded frames without copying the entire encoded object."""

    def __init__(self, tensor):
        self.view = memoryview(tensor.numpy())
        self.offset = 0

    def read(self, size=-1):
        end = len(self.view) if size < 0 else min(len(self.view), self.offset + size)
        result = self.view[self.offset : end].tobytes()
        self.offset = end
        return result


def validate_wire_metadata(
    meta, *, shape, dtype, max_decoded_bytes=DEFAULT_MAX_DECODED_BYTES
):
    """Validate before allocating either wire or decoded tensors."""
    if not isinstance(meta, dict) or meta.get("encoding") not in WIRE_ENCODINGS:
        raise ValueError("Unsupported capture wire encoding")
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) > 32
        or any(type(d) is not int or d < 0 for d in shape)
        or not isinstance(dtype, str)
    ):
        raise ValueError("Invalid decoded capture tensor specification")
    tensor_dtype = getattr(torch, dtype, None)
    if not isinstance(tensor_dtype, torch.dtype):
        raise ValueError("Unsupported decoded capture dtype")
    decoded_bytes = math.prod(shape) * tensor_dtype.itemsize
    if decoded_bytes > max_decoded_bytes:
        raise ValueError("Capture wire tensor exceeds decoded byte limit")
    if (
        type(meta.get("decoded_bytes")) is not int
        or meta["decoded_bytes"] != decoded_bytes
    ):
        raise ValueError(
            "Capture wire decoded length differs from tensor specification"
        )
    encoded_bytes = meta.get("encoded_bytes")
    if type(
        encoded_bytes
    ) is not int or not 0 < encoded_bytes <= decoded_bytes + decoded_bytes // 8 + (
        1 << 20
    ):
        raise ValueError("Capture wire encoded length exceeds bounded record size")
    if meta["encoding"] == WIRE_ENCODING and (
        not isinstance(meta.get("sha256"), str)
        or not re.fullmatch("[0-9a-f]{64}", meta["sha256"])
    ):
        raise ValueError("Invalid capture wire checksum")
    if meta["encoding"] == NO_HASH_WIRE_ENCODING and "sha256" in meta:
        raise ValueError("Hash-free capture must not carry a checksum")
    return encoded_bytes


def encode_wire_tensor(tensor, *, level=1, encoding=WIRE_ENCODING):
    """Return CPU uint8 wire storage and its explicit encoding metadata."""
    if tensor.device.type != "cpu":
        raise ValueError("Stage capture tensor on CPU before wire compression")
    if encoding not in WIRE_ENCODINGS:
        raise ValueError("Unsupported capture wire encoding")
    checksums = encoding == WIRE_ENCODING
    stream = io.BytesIO()
    save_zstd_features({"tensor": tensor}, stream, level=level, checksums=checksums)
    # A writable BytesIO memoryview owns the buffer for the tensor lifetime.
    # Avoid getvalue()/bytes/clone copies of a potentially multi-GiB object.
    encoded = torch.frombuffer(stream.getbuffer(), dtype=torch.uint8)
    meta = {
        "encoding": encoding,
        "encoded_bytes": encoded.numel(),
        "decoded_bytes": tensor.numel() * tensor.element_size(),
    }
    if checksums:
        meta["sha256"] = hashlib.sha256(memoryview(encoded.numpy())).hexdigest()
    validate_wire_metadata(
        meta, shape=list(tensor.shape), dtype=str(tensor.dtype).removeprefix("torch.")
    )
    return encoded, meta


def decode_wire_tensor(
    encoded, meta, *, shape, dtype, max_decoded_bytes=DEFAULT_MAX_DECODED_BYTES
):
    nbytes = validate_wire_metadata(
        meta, shape=shape, dtype=dtype, max_decoded_bytes=max_decoded_bytes
    )
    if (
        encoded.device.type != "cpu"
        or encoded.dtype != torch.uint8
        or encoded.ndim != 1
        or encoded.numel() != nbytes
    ):
        raise ValueError(
            "Capture wire payload does not match declared uint8 byte length"
        )
    checksums = meta["encoding"] == WIRE_ENCODING
    if (
        checksums
        and hashlib.sha256(memoryview(encoded.numpy())).hexdigest() != meta["sha256"]
    ):
        raise ValueError("Capture wire checksum mismatch")
    record = load_zstd_features(
        _BufferReader(encoded),
        max_record_bytes=meta["decoded_bytes"],
        expected_version=1 if checksums else 2,
    )
    if set(record) != {"tensor"}:
        raise ValueError("Capture wire record must contain exactly one tensor")
    tensor = record["tensor"]
    if (
        tuple(tensor.shape) != tuple(shape)
        or str(tensor.dtype).removeprefix("torch.") != dtype
    ):
        raise ValueError("Decoded capture tensor shape/dtype differs from FeatureSpec")
    return tensor
