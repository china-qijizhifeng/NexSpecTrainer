"""Autograd bindings to TileKernels; no eager numerical fallback.

The wide, FP32 mHC RMS uses Transformer Engine's fused RMSNorm (the Megatron
backend): TileKernels' standalone RMS backward is limited to 8192 columns.
"""

import torch


def require_cuda(x):
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device)[0] != 9:
        raise RuntimeError("Native V4.1 training requires Hopper SM90 CUDA kernels")


class RMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        require_cuda(x)
        from tile_kernels.quant import norm_forward

        y, rstd, saved = norm_forward(x, weight, eps, with_rstd=True)
        ctx.save_for_backward(saved, weight, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        from tile_kernels.quant import norm_backward

        x, weight, rstd = ctx.saved_tensors
        dx, dw = norm_backward(dy.contiguous(), x, weight, rstd)
        return dx, dw, None


class WideRMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps):
        require_cuda(x)
        from transformer_engine.pytorch.constants import TE_DType
        from transformer_engine_torch import rmsnorm_fwd

        weight = torch.ones(x.shape[-1], device=x.device, dtype=torch.float32)
        y, _, rstd = rmsnorm_fwd(
            x, weight, eps, None, None, TE_DType[x.dtype], 0, False
        )
        ctx.save_for_backward(x, weight, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        from transformer_engine_torch import rmsnorm_bwd

        x, weight, rstd = ctx.saved_tensors
        dx, _ = rmsnorm_bwd(dy.contiguous(), x, rstd, weight, 0, False)
        return dx, None


class Rotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, positions, cache, rotary_dim, inverse):
        from tile_kernels.transform import apply_rotary

        require_cuda(x)
        y = x.clone(memory_format=torch.contiguous_format)
        apply_rotary(
            y[..., -rotary_dim:],
            cache,
            positions=positions,
            interleaved=True,
            conjugate=inverse,
        )
        ctx.save_for_backward(positions, cache)
        ctx.rotary_dim, ctx.inverse = rotary_dim, inverse
        return y

    @staticmethod
    def backward(ctx, dy):
        from tile_kernels.transform import apply_rotary

        positions, cache = ctx.saved_tensors
        dx = dy.clone(memory_format=torch.contiguous_format)
        apply_rotary(
            dx[..., -ctx.rotary_dim :],
            cache,
            positions=positions,
            interleaved=True,
            conjugate=not ctx.inverse,
        )
        return dx, None, None, None, None


class SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, limit):
        from tile_kernels.quant import swiglu_forward

        ctx.save_for_backward(x, weights)
        ctx.limit = limit
        return swiglu_forward(x, "bf16", topk_weights=weights, clamp_value=limit)

    @staticmethod
    def backward(ctx, dy):
        from tile_kernels.quant import swiglu_backward

        x, weights = ctx.saved_tensors
        result = swiglu_backward(
            x, dy.contiguous(), "bf16", topk_weights=weights, clamp_value=ctx.limit
        )
        return result[0], result[1] if weights is not None else None, None
