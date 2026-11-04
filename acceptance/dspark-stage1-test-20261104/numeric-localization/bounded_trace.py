"""Bounded tensor-row digests for a future diagnostic; no SGLang hooks installed."""

import hashlib
import json
import os
from pathlib import Path

import torch

MAX_ROW_BYTES = 2 << 20
MAX_LINE_BYTES = 4 << 10
MAX_TRACE_BYTES = 8 << 20
MAX_EVENTS = 16000


def tensor_row_digest(tensor, row=-1):
    """Slice before any copy; hash exact bytes without changing the producer."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected an existing tensor, not a deferred output")
    if tensor.layout != torch.strided or tensor.ndim > 3 or tensor.numel() == 0:
        raise ValueError("unsupported tensor layout, dimension or empty tensor")
    if tensor.ndim >= 2:
        selected = tensor[row]
    else:
        selected = tensor
    size = selected.numel() * selected.element_size()
    if size > MAX_ROW_BYTES:
        raise ValueError("tensor row exceeds2MiB bound before CPU copy")
    cpu = selected.detach().to(device="cpu", copy=True).contiguous()
    if not bool(torch.isfinite(cpu).all()):
        raise ValueError("non-finite tensor row")
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    assert len(raw) == size
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "row": row if tensor.ndim >= 2 else None,
        "selected_shape": list(selected.shape),
        "selected_bytes": size,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "finite": True,
    }


class TraceWriter:
    """One exclusive private JSONL stream; bounds reject instead of truncating."""

    def __init__(self, path):
        self.path = Path(path)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.stream = os.fdopen(descriptor, "wb")
        self.events = 0
        self.bytes_written = 0

    def write(self, event):
        encoded = (json.dumps(event, allow_nan=False, sort_keys=True) + "\n").encode()
        if len(encoded) > MAX_LINE_BYTES:
            raise ValueError("event exceeds4KiB bound")
        if self.events >= MAX_EVENTS:
            raise ValueError("event count bound exceeded")
        if self.bytes_written + len(encoded) > MAX_TRACE_BYTES:
            raise ValueError("trace exceeds8MiB bound")
        self.stream.write(encoded)
        self.stream.flush()
        self.events += 1
        self.bytes_written += len(encoded)

    def close(self):
        if not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
