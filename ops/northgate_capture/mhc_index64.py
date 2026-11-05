# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.
# Derived from sglang 38d9e4a3, kernels/ops/layernorm/mhc.py.
"""Preserve MHC arithmetic while promoting wide residual row offsets to int64."""
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def _mhc_combine_kernel(
    x_ptr,
    pre_ptr,
    y_ptr,
    H,
    x_stride_m,
    pre_stride_m,
    pre_stride_k,
    y_stride_m,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs_h < H
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(HC):
        pk = tl.load(pre_ptr + pid_m * pre_stride_m + k * pre_stride_k).to(tl.float32)
        xv = tl.load(
            x_ptr + pid_m * x_stride_m + k * H + offs_h, mask=mask, other=0.0
        ).to(tl.float32)
        acc += pk * xv
    tl.store(y_ptr + pid_m * y_stride_m + offs_h, acc, mask=mask)

@triton.jit
def _hc_mix_stats_partial_kernel(
    x_ptr,
    w_ptr,
    part_mix_ptr,
    part_sq_ptr,
    M,
    K,
    x_stride_m,
    w_stride_n,
    MIX: tl.constexpr,
    MIX_PAD: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """Mixing dot products and row sum of squares over one K slice."""
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = tl.arange(0, MIX_PAD)
    mask_m = offs_m < M
    mask_n = offs_n < MIX
    k_per_slice = K // NUM_SLICES
    k_start = pid_s * k_per_slice
    acc = tl.zeros([BLOCK_M, MIX_PAD], dtype=tl.float32)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for kb in range(0, k_per_slice, BLOCK_K):
        offs_k = k_start + kb + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_start + k_per_slice
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * x_stride_m + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        w_tile = tl.load(
            w_ptr + offs_n[None, :] * w_stride_n + offs_k[:, None],
            mask=mask_n[None, :] & mask_k[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(x_tile, w_tile, input_precision=DOT_PRECISION)
        sq += tl.sum(x_tile * x_tile, axis=1)
    tl.store(
        part_mix_ptr + (pid_s * M + offs_m[:, None]) * MIX + offs_n[None, :],
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )
    tl.store(part_sq_ptr + pid_s * M + offs_m, sq, mask=mask_m)

@triton.jit
def _engram_gate_kernel(
    X,
    KV,
    QW,
    KW,
    IDS,
    O,
    IMAGE_ID,
    D: tl.constexpr,
    HC: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP: tl.constexpr,
    B: tl.constexpr,
    HAS_IDS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    token = row // HC
    hc = row % HC
    col = tl.arange(0, B)
    mask = col < D
    x = tl.load(X + row * D + col, mask, 0).to(tl.float32)
    key = tl.load(KV + token * (HC + 1) * D + hc * D + col, mask, 0).to(tl.float32)
    q_weight = tl.load(QW + hc * D + col, mask, 0).to(tl.float32)
    k_weight = tl.load(KW + hc * D + col, mask, 0).to(tl.float32)
    weight = q_weight * k_weight
    rstd = tl.rsqrt(tl.sum(x * x, 0) / D + EPS) * tl.rsqrt(
        tl.sum(key * key, 0) / D + EPS
    )
    dot = tl.sum((x * weight) * key, 0) * rstd * (D**-0.5)
    gate = tl.sigmoid(libdevice.copysign(tl.sqrt(tl.maximum(tl.abs(dot), CLAMP)), dot))
    value = tl.load(KV + token * (HC + 1) * D + HC * D + col, mask, 0).to(tl.float32)
    out = x + gate * value
    if HAS_IDS:
        out = tl.where(tl.load(IDS + token) == IMAGE_ID, x, out)
    tl.store(O + row * D + col, out, mask)

def install():
    from sglang.kernels.ops.layernorm import mhc
    from sglang.kernels.ops.embeddings import engram_gate

    engram_gate._engram_gate_kernel = _engram_gate_kernel

    mhc._mhc_combine_kernel = _mhc_combine_kernel
    mhc._hc_mix_stats_partial_kernel = _hc_mix_stats_partial_kernel
