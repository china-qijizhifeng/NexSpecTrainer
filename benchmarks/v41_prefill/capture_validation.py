"""Full shape, finite-value and alignment checks without content fingerprints."""

import torch


def forbid_capture_hashes():
    """Fail the experiment if either capture codec attempts SHA256."""
    from types import SimpleNamespace

    from specforge.data import feature_file, wire_codec

    def forbidden(*args, **kwargs):
        raise AssertionError("Hashing is forbidden in this capture experiment")

    guard = SimpleNamespace(sha256=forbidden)
    feature_file.hashlib = guard
    wire_codec.hashlib = guard


def validate(tensors, row):
    n = len(row["input_ids"])
    shapes = {
        "input_ids": (1, n),
        "loss_mask": (1, n),
        "hidden_states": (1, n, 15360),
        "target_last_hidden_states": (1, n, 5120),
    }
    if set(tensors) != set(shapes):
        raise ValueError("Unexpected capture feature names")
    for name, shape in shapes.items():
        value = tensors[name]
        if value.device.type != "cpu" or tuple(value.shape) != shape:
            raise ValueError(f"{name}: expected CPU {shape}, got {value.shape}")
        if name in ("input_ids", "loss_mask"):
            if value.dtype != torch.int64 or not torch.equal(
                value[0], torch.tensor(row[name], dtype=torch.int64)
            ):
                raise ValueError(f"{name}: capture alignment mismatch")
        elif value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
            raise ValueError(f"{name}: finite BF16 capture required")
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in tensors.items()
    }
