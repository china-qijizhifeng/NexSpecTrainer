# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
# Adapted from SGLang 38d9e4a3, kernels/ops/layernorm/mhc.py.
"""Keep the mHC compensated projection's tile geometry fixed on Hopper."""


def mix_stats_bf16x3(
    x, weight_parts, scale, base, sinkhorn_iters, rms_eps, hc_eps, hc_mult=4
):
    import torch
    import triton
    from sglang.kernels.ops.layernorm import mhc

    m, k = x.shape
    mix = (2 + hc_mult) * hc_mult
    slices = mhc._HC_MIX_COMPENSATED_SLICES
    # The 64-row Hopper variant is not repeatable for the captured 1784-row
    # CP operand. Use the reference prefill's 128-row tile at every row count.
    block_m = 128
    assert x.is_contiguous() and x.dtype == torch.bfloat16
    assert k % (slices * mhc._HC_MIX_BLOCK_K) == 0
    part_mix = torch.empty((slices, m, mix), device=x.device, dtype=torch.float32)
    sq = torch.empty((slices, m), device=x.device, dtype=torch.float32)
    pre = torch.empty((m, hc_mult), device=x.device, dtype=torch.float32)
    post = torch.empty_like(pre)
    comb = torch.empty((m, hc_mult, hc_mult), device=x.device, dtype=torch.float32)
    mhc._hc_mix_stats_bf16x3_kernel[(triton.cdiv(m, block_m), slices)](
        x,
        *weight_parts,
        part_mix,
        sq,
        m,
        K=k,
        K_PER_SLICE=k // slices,
        MIX_COLS=mix,
        MIX_PAD=triton.next_power_of_2(mix),
        BLOCK_K=mhc._HC_MIX_BLOCK_K,
        BLOCK_M=block_m,
        num_warps=4,
        num_stages=3,
    )
    mhc._hc_mix_reduce_sinkhorn_kernel[(m,)](
        part_mix,
        sq,
        scale,
        base,
        pre,
        post,
        comb,
        m,
        1.0 / k,
        rms_eps,
        MIX=mix,
        HC=hc_mult,
        NUM_SLICES=slices,
        ITERS=sinkhorn_iters,
        EPS=hc_eps,
        num_warps=1,
    )
    return pre, post, comb
