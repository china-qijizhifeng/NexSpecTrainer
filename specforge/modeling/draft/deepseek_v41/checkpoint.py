"""Selective native checkpoint I/O and FP8 / packed MXFP4 conversion.

Only requested shards are fetched. Dequantization is for training; exporting
re-quantizes to the original tensor layout and preserves the target checkpoint.
"""

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

FP4_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


class NativeCheckpoint:
    def __init__(self, source, cache_dir=None):
        self.source, self.cache_dir = str(source), cache_dir
        self.index = json.loads(self.file("model.safetensors.index.json").read_text())
        self.weight_map = self.index["weight_map"]

    def file(self, name):
        root = Path(self.source)
        if root.is_dir():
            return root / name
        return Path(hf_hub_download(self.source, name, cache_dir=self.cache_dir))

    def tensor(self, key):
        with safe_open(
            self.file(self.weight_map[key]), framework="pt", device="cpu"
        ) as f:
            return f.get_tensor(key)

    def floating(self, key):
        raw = self.tensor(key)
        scale_key = key.removesuffix(".weight") + ".scale"
        scale = (
            self.tensor(scale_key)
            if key.endswith(".weight") and scale_key in self.weight_map
            else None
        )
        return dequantize(raw, scale)


def dequantize(weight, scale):
    if scale is None:
        if not weight.is_floating_point():
            raise ValueError("integer native weight without scale")
        return weight.float()
    scales = scale.float()
    if weight.dtype in (torch.int8, torch.uint8):
        packed = weight.view(torch.uint8)
        indices = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
        lut = torch.tensor(FP4_VALUES, device=weight.device)
        values = lut[indices]
        if scales.shape != (values.shape[0], values.shape[1] // 32):
            raise ValueError(
                "native MXFP4 requires one E8M0 scale per row / 32 columns"
            )
        return values * scales.repeat_interleave(32, -1)
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"unsupported scaled native dtype {weight.dtype}")
    if scales.shape != ((weight.shape[0] + 31) // 32, (weight.shape[1] + 31) // 32):
        raise ValueError("native FP8 requires 32 x 32 block scales")
    return (
        weight.float()
        * scales.repeat_interleave(32, 0).repeat_interleave(32, 1)[
            : weight.shape[0], : weight.shape[1]
        ]
    )


def quantize_like(value, template, scale_template=None):
    """Return a tensor and optional scale in the source checkpoint's format."""
    value = value.detach().float().cpu()
    if not torch.isfinite(value).all():
        raise ValueError("cannot export non-finite native weights")
    if scale_template is None:
        return value.to(template.dtype), None
    if template.dtype in (torch.int8, torch.uint8):
        groups = value.reshape(value.shape[0], -1, 32)
        scale = _power_two_scale(groups.abs().amax(-1), 6.0)
        normalized = (groups / scale[..., None]).flatten(1)
        # Nearest magnitude; ties use torch.argmin's first (smaller) entry.
        lut = torch.tensor(FP4_VALUES[:8])
        # Chunk rows: a full [experts * rows, cols, 8] temporary is excessive.
        codes = torch.empty_like(normalized, dtype=torch.uint8)
        for start in range(0, normalized.shape[0], 128):
            block = normalized[start : start + 128]
            codes[start : start + 128] = (
                (block.abs()[..., None] - lut).abs().argmin(-1).to(torch.uint8)
            )
        codes = codes | ((normalized < 0).to(torch.uint8) << 3)
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        return packed.view(template.dtype), scale.to(scale_template.dtype)
    if template.dtype == torch.float8_e4m3fn:
        rows, cols = value.shape
        padded = torch.nn.functional.pad(value, (0, (-cols) % 32, 0, (-rows) % 32))
        blocks = padded.reshape(padded.shape[0] // 32, 32, padded.shape[1] // 32, 32)
        scale = _power_two_scale(blocks.abs().amax((1, 3)), 448.0)
        expanded = scale.repeat_interleave(32, 0).repeat_interleave(32, 1)[:rows, :cols]
        return (value / expanded).clamp(-448, 448).to(template.dtype), scale.to(
            scale_template.dtype
        )
    raise ValueError(f"unsupported native export dtype {template.dtype}")


def _power_two_scale(maximum, bound):
    return torch.pow(
        2.0,
        torch.ceil(torch.log2((maximum / bound).clamp_min(2.0**-126))).clamp(-126, 127),
    )


def native_parts(key, shape):
    """Map a training parameter to native keys and slices (no full-state copy)."""
    if key.endswith(".ffn.gate_up"):
        prefix = key.removesuffix("gate_up")
        mid = shape[1] // 2
        return [
            (f"{prefix}experts.{i}.{part}.weight", (i, sl))
            for i in range(shape[0])
            for part, sl in (("w1", slice(0, mid)), ("w3", slice(mid, None)))
        ]
    if key.endswith(".ffn.down"):
        prefix = key.removesuffix("down")
        return [(f"{prefix}experts.{i}.w2.weight", i) for i in range(shape[0])]
    for kind in ("attn", "ffn"):
        for part in ("fn", "base", "scale"):
            key = key.replace(f".hc_{kind}.{part}", f".hc_{kind}_{part}")
    key = key.replace(".attn.sink.weight", ".attn.attn_sink")
    return [(key, Ellipsis)]


def initialize_native(
    model, source, *, device="cpu", dtype=torch.bfloat16, cache_dir=None
):
    checkpoint = NativeCheckpoint(source, cache_dir)
    parameters = dict(model.named_parameters())
    expected = {
        native
        for key, p in parameters.items()
        for native, _ in native_parts(key, p.shape)
    }
    actual = {
        k
        for k in checkpoint.weight_map
        if k.startswith("mtp.") and not k.endswith(".scale")
    }
    if actual != expected:
        raise ValueError(
            f"native MTP key mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    for key, parameter in parameters.items():
        parent, name = key.rsplit(".", 1)
        module = model.get_submodule(parent)
        pieces = parent.split(".")
        keep_fp32 = any(
            getattr(
                model.get_submodule(".".join(pieces[:i])), "_specforge_keep_fp32", False
            )
            for i in range(1, len(pieces) + 1)
        )
        value = torch.empty(
            parameter.shape, device=device, dtype=torch.float32 if keep_fp32 else dtype
        )
        for native, index in native_parts(key, parameter.shape):
            loaded = checkpoint.floating(native)
            if loaded.shape != value[index].shape:
                raise ValueError(
                    f"native {native}: expected {tuple(value[index].shape)}, got {tuple(loaded.shape)}"
                )
            value[index].copy_(loaded)
        module._parameters[name] = torch.nn.Parameter(
            value, requires_grad=parameter.requires_grad
        )
    return model
