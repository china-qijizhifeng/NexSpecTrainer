"""V4.1 modules backed by TileKernels, DeepGEMM and Megatron fused dispatch.

PyTorch supplies parameters, shape/index bookkeeping and cuBLAS linear layers;
normalization, routing, activations and mHC execute explicit GPU kernels.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .kernels.elementwise import RMS, Rotary, SwiGLU, WideRMS
from .kernels.mhc import PostMix, PreMix, Sinkhorn, SplitMixes
from .kernels.moe import GroupedLinear, Route


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return RMS.apply(
            x.reshape(-1, x.shape[-1]).contiguous(), self.weight.to(x.dtype), self.eps
        ).reshape_as(x)


def rotary(x, positions, cache, rotary_dim, inverse=False):
    return Rotary.apply(x, positions.contiguous(), cache, rotary_dim, inverse)


def hc_mixes(x, weight, scale, bias, iterations, eps, norm_eps):
    h = x.shape[-2]
    flat = x.flatten(-2).reshape(-1, weight.shape[1]).float()
    normalized = WideRMS.apply(flat, norm_eps)
    mixes = F.linear(normalized, weight.float())
    pre, post, comb = SplitMixes.apply(mixes, scale.float(), bias.float(), h, eps)
    return (
        pre.reshape(*x.shape[:-2], h),
        post.reshape(*x.shape[:-2], h),
        Sinkhorn.apply(comb, iterations, eps).reshape(*x.shape[:-2], h, h),
    )


class HCMappings(nn.Module):
    _specforge_keep_fp32 = True

    def __init__(self, c):
        super().__init__()
        h, d = c.hc_mult, c.hidden_size
        self.fn = nn.Parameter(torch.zeros((2 + h) * h, h * d))
        self.base = nn.Parameter(torch.zeros((2 + h) * h))
        self.scale = nn.Parameter(torch.ones(3))
        self.c = c

    def forward(self, x):
        c = self.c
        return hc_mixes(
            x,
            self.fn,
            self.scale,
            self.base,
            c.hc_sinkhorn_iters,
            c.hc_eps,
            c.rms_norm_eps,
        )


def hc_pre(x, pre):
    return PreMix.apply(
        x.reshape(-1, *x.shape[-2:]), pre.reshape(-1, pre.shape[-1])
    ).reshape(*x.shape[:-2], x.shape[-1])


def hc_post(x, residual, post, comb):
    return PostMix.apply(x, residual, post.unsqueeze(-1), comb)


class Expert(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w1 = nn.Linear(c.hidden_size, c.moe_intermediate_size, bias=False)
        self.w2 = nn.Linear(c.moe_intermediate_size, c.hidden_size, bias=False)
        self.w3 = nn.Linear(c.hidden_size, c.moe_intermediate_size, bias=False)
        self.limit = c.swiglu_limit

    def forward(self, x):
        gate_up = torch.cat((self.w1(x), self.w3(x)), -1).contiguous()
        return self.w2(SwiGLU.apply(gate_up, None, self.limit))


class Gate(nn.Module):
    _specforge_keep_fp32 = True

    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c.n_routed_experts, c.hidden_size))
        nn.init.normal_(self.weight, std=0.02)
        self.bias = nn.Parameter(torch.zeros(c.n_routed_experts), requires_grad=False)
        self.bias_vl = nn.Parameter(
            torch.zeros(c.n_routed_experts), requires_grad=False
        )
        self.topk, self.scale = c.num_experts_per_tok, c.routed_scaling_factor

    def forward(self, x):
        return Route.apply(
            F.linear(x.float(), self.weight.float()),
            self.bias.float(),
            self.topk,
            self.scale,
        )


class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = Gate(c)
        # Store expert matrices already stacked. No per-step weight stacking or
        # Python expert loop; checkpoint I/O translates to/from native keys.
        self.gate_up = nn.Parameter(
            torch.empty(c.n_routed_experts, 2 * c.moe_intermediate_size, c.hidden_size)
        )
        self.down = nn.Parameter(
            torch.empty(c.n_routed_experts, c.hidden_size, c.moe_intermediate_size)
        )
        self.shared_experts = Expert(c)
        self.limit = c.swiglu_limit

    def forward(self, x):
        import deep_gemm
        from megatron.core.transformer.moe.moe_utils import permute, unpermute

        flat = x.reshape(-1, x.shape[-1])
        weights, indices = self.gate(flat)
        routing_map = torch.zeros(
            flat.shape[0], self.gate_up.shape[0], device=x.device, dtype=torch.bool
        )
        routing_map.scatter_(1, indices, True)
        probs = torch.zeros_like(routing_map, dtype=torch.float32).scatter(
            1, indices, weights
        )
        counts = routing_map.sum(0)
        alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
        if alignment % 128:
            raise RuntimeError(
                "DeepGEMM K-grouped backward requires token alignment divisible by 128"
            )
        packed, pweights, rows, padding, padded_counts = permute(
            flat,
            routing_map,
            probs=probs,
            fused=True,
            tokens_per_expert=counts,
            align_size=alignment,
        )
        ends = padded_counts.cumsum(0, dtype=torch.int32)
        counts_cpu = padded_counts.cpu().tolist()
        up = GroupedLinear.apply(packed, self.gate_up, ends, counts_cpu)
        activated = SwiGLU.apply(up, pweights, self.limit)
        out = GroupedLinear.apply(activated, self.down, ends, counts_cpu)
        out = unpermute(out, rows, flat.shape, fused=True, pad_offsets=padding)
        return (out + self.shared_experts(flat)).reshape_as(x)
