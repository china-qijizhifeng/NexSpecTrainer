"""DeepGEMM BF16 grouped GEMM autograd and TileKernels router.

Aligned, expert-major inputs come from Megatron's TE fused permutation. The
same padded prefix sums are used for forward, dgrad and K-grouped wgrad.
"""

import torch

from .elementwise import require_cuda


def grouped_weight_gradient(dy, x, weight, counts_cpu, *, expert_chunk=8):
    """Accumulate each expert in FP32, with bounded temporary storage.

    Experts are independent reductions. Splitting only the expert dimension
    preserves each expert's complete token reduction and final BF16 rounding.
    The native up matrix otherwise needs 11.25 GiB FP32 plus 5.625 GiB BF16
    simultaneously, on top of FSDP's accumulated full gradients.
    """
    import deep_gemm

    counts = torch.tensor(counts_cpu, device=x.device, dtype=torch.int32)
    gradient = torch.empty_like(weight)
    workspace = torch.empty(
        (min(expert_chunk, weight.shape[0]), *weight.shape[1:]),
        device=weight.device,
        dtype=torch.float32,
    )
    row = 0
    for start in range(0, weight.shape[0], expert_chunk):
        stop = min(start + expert_chunk, weight.shape[0])
        lengths = counts_cpu[start:stop]
        end = row + sum(lengths)
        partial = workspace[: stop - start]
        partial.zero_()
        if end > row:
            deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
                dy[row:end], x[row:end], partial, lengths, counts[start:stop], partial
            )
        gradient[start:stop].copy_(partial)
        row = end
    return gradient


class GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, ends, counts_cpu):
        import deep_gemm

        require_cuda(x)
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise TypeError(
                "DeepGEMM native draft uses BF16 activations and expert weights"
            )
        out = x.new_empty(x.shape[0], weight.shape[1])
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            x, weight, out, ends, use_psum_layout=True
        )
        ctx.save_for_backward(x, weight, ends)
        ctx.counts_cpu = counts_cpu
        return out

    @staticmethod
    def backward(ctx, dy):
        import deep_gemm

        x, weight, ends = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        deep_gemm.m_grouped_bf16_gemm_nn_contiguous(
            dy, weight, dx, ends, use_psum_layout=True
        )
        # Hopper wgrad requires aligned host K lengths and FP32 accumulation.
        # M-grouped forward/dgrad accept prefix sums; K-grouped uses lengths.
        dw = grouped_weight_gradient(dy, x, weight, ctx.counts_cpu)
        return dx, dw, None, None


class Route(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, bias, topk, scale):
        from tile_kernels.moe import moe_topk_gate_forward

        scores = torch.empty_like(logits)
        indices, weights = moe_topk_gate_forward(
            logits, topk, False, 0, scale, 0, bias=bias, scores=scores
        )
        ctx.save_for_backward(scores, indices, weights)
        ctx.scale = scale
        ctx.mark_non_differentiable(indices)
        return weights, indices

    @staticmethod
    def backward(ctx, dweights, _):
        from tile_kernels.moe import moe_topk_gate_backward

        return (
            moe_topk_gate_backward(
                *ctx.saved_tensors, dweights.contiguous(), ctx.scale
            ),
            None,
            None,
            None,
        )
