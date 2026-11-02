"""Differentiable V4.1 reference operators.

Adapted from DeepSeek-V4.1-Flash inference/model.py and kernel.py (MIT).
See NOTICE.deepseek-v41.md for the pinned upstream and license.
"""

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        y = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        )
        return (y * self.weight.float()).to(x.dtype)


def rotary(x, positions, rotary_dim, theta, inverse=False):
    """Adjacent real/imaginary pairs; pure SWA uses base RoPE, without YaRN."""
    freq = theta ** (
        -torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32)
        / rotary_dim
    )
    angle = positions.float()[..., None] * freq
    while angle.ndim < x.ndim:
        angle = angle.unsqueeze(-2)
    if inverse:
        angle = -angle
    tail = x[..., -rotary_dim:].float().unflatten(-1, (-1, 2))
    real, imag = tail.unbind(-1)
    rotated = torch.stack(
        (
            real * angle.cos() - imag * angle.sin(),
            real * angle.sin() + imag * angle.cos(),
        ),
        -1,
    ).flatten(-2)
    return torch.cat((x[..., :-rotary_dim], rotated.to(x.dtype)), -1)


def hc_mixes(x, weight, scale, bias, iterations, eps, norm_eps):
    h = x.shape[-2]
    flat = x.flatten(-2).float()
    mix = F.linear(flat, weight.float()) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + norm_eps
    )
    pre = torch.sigmoid(mix[..., :h] * scale[0].float() + bias[:h].float()) + eps
    post = 2 * torch.sigmoid(
        mix[..., h : 2 * h] * scale[1].float() + bias[h : 2 * h].float()
    )
    comb = (mix[..., 2 * h :] * scale[2].float() + bias[2 * h :].float()).unflatten(
        -1, (h, h)
    )
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre(x, pre):
    return (pre.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype)


def hc_post(x, residual, post, comb):
    # comb[source, destination], as in the released kernel/model.
    mixed = torch.einsum("...ij,...id->...jd", comb.float(), residual.float())
    return (post.unsqueeze(-1) * x.float().unsqueeze(-2) + mixed).to(x.dtype)


class Expert(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w1 = nn.Linear(c.hidden_size, c.moe_intermediate_size, bias=False)
        self.w2 = nn.Linear(c.moe_intermediate_size, c.hidden_size, bias=False)
        self.w3 = nn.Linear(c.hidden_size, c.moe_intermediate_size, bias=False)
        self.limit = c.swiglu_limit

    def forward(self, x, weight=None):
        gate, up = self.w1(x).float(), self.w3(x).float()
        if self.limit > 0:
            gate, up = gate.clamp(max=self.limit), up.clamp(-self.limit, self.limit)
        y = F.silu(gate) * up
        if weight is not None:
            y = y * weight
        return self.w2(y.to(x.dtype))


class Gate(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c.n_routed_experts, c.hidden_size))
        nn.init.normal_(self.weight, std=0.02)
        # Correction biases select experts; no gradient through discrete top-k.
        self.bias = nn.Parameter(torch.zeros(c.n_routed_experts), requires_grad=False)
        self.bias_vl = nn.Parameter(
            torch.zeros(c.n_routed_experts), requires_grad=False
        )
        self.topk, self.scale = c.num_experts_per_tok, c.routed_scaling_factor

    def forward(self, x):
        scores = F.softplus(F.linear(x.float(), self.weight.float())).sqrt()
        indices = (scores + self.bias.float()).topk(self.topk, dim=-1).indices
        weights = scores.gather(-1, indices)
        if self.topk > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return weights * self.scale, indices


class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = Gate(c)
        self.experts = nn.ModuleList([Expert(c) for _ in range(c.n_routed_experts)])
        self.shared_experts = Expert(c)

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        weights, indices = self.gate(flat)
        out = torch.zeros_like(flat, dtype=torch.float32)
        # Empty expert batches are deliberate: every expert participates in
        # autograd/FSDP, including ranks whose local tokens do not route to it.
        for i, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == i)
            values = expert(flat[rows], weights[rows, slots, None]).float()
            out = out.index_add(0, rows, values)
        return (out + self.shared_experts(flat).float()).to(x.dtype).reshape(shape)
