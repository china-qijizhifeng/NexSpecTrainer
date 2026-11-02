"""Numerical forward/backward acceptance; requires the pinned Hopper stack."""

import pytest
import torch
from torch.nn import functional as F

from specforge.modeling.draft.deepseek_v41.kernels.elementwise import (
    Rotary,
    SwiGLU,
    WideRMS,
)
from specforge.modeling.draft.deepseek_v41.kernels.moe import GroupedLinear
from specforge.modeling.draft.deepseek_v41.ops import (
    MoE,
    RMSNorm,
    hc_mixes,
    hc_post,
    hc_pre,
)

from . import reference
from .test_checkpoint import tiny_config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Hopper GPU acceptance suite"
)


def setup_module():
    if torch.cuda.is_available():
        from specforge.modeling.draft.deepseek_v41.validation import preflight

        preflight()


def compare_gradients(actual, expected, inputs, refs, tolerance=0.04):
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=tolerance, atol=tolerance
    )
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for a, b in zip(inputs, refs):
        assert a.grad is not None and torch.isfinite(a.grad).all()
        torch.testing.assert_close(
            a.grad.float(), b.grad.float(), rtol=tolerance, atol=tolerance
        )


def clone(x):
    return x.detach().clone().requires_grad_(True)


@pytest.mark.parametrize("width", [512, 5120, 20480])
def test_rms(width):
    x = torch.randn(
        8,
        width,
        device="cuda",
        dtype=torch.float32 if width > 8192 else torch.bfloat16,
        requires_grad=True,
    )
    ref = clone(x)
    if width > 8192:
        y = WideRMS.apply(x, 1e-20)
        want = ref * torch.rsqrt(ref.square().mean(-1, keepdim=True) + 1e-20)
        compare_gradients(y, want, [x], [ref])
    else:
        norm = RMSNorm(width, 1e-20).cuda().bfloat16()
        refnorm = reference.RMSNorm(width, 1e-20).cuda().bfloat16()
        compare_gradients(
            norm(x), refnorm(ref), [x, norm.weight], [ref, refnorm.weight]
        )


def test_mhc_previous_sublayer_and_orientation():
    x = torch.randn(
        2, 5, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    w = (torch.randn(24, 512, device="cuda") / 20).requires_grad_()
    scale = torch.tensor([0.4, 0.7, 0.8], device="cuda", requires_grad=True)
    bias = torch.randn(24, device="cuda", requires_grad=True)
    previous = torch.rand(2, 5, 4, device="cuda", requires_grad=True)
    args = [x, w, scale, bias, previous]
    refs = [clone(a) for a in args]
    pre, post, comb = hc_mixes(x, w, scale, bias, 20, 1e-6, 1e-20)
    rp, rpost, rc = reference.hc_mixes(*refs[:4], 20, 1e-6, 1e-20)
    out = hc_post(hc_pre(x, previous), x, post, comb)
    expected = reference.hc_post(reference.hc_pre(refs[0], refs[4]), refs[0], rpost, rc)
    compare_gradients(hc_pre(out, pre), reference.hc_pre(expected, rp), args, refs)


def test_swiglu_clamp_and_weight_grad():
    x = (
        torch.randn(32, 256, device="cuda", dtype=torch.bfloat16) * 12
    ).requires_grad_()
    weights = torch.rand(32, device="cuda", requires_grad=True)
    rx, rw = clone(x), clone(weights)
    gate, up = rx.float().chunk(2, -1)
    expected = (F.silu(gate.clamp(max=10)) * up.clamp(-10, 10) * rw[:, None]).bfloat16()
    compare_gradients(
        SwiGLU.apply(x, weights, 10.0), expected, [x, weights], [rx, rw], 0.5
    )


@pytest.mark.parametrize("inverse", [False, True])
def test_rope(inverse):
    x = torch.randn(
        2, 5, 4, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    ref = clone(x)
    positions = torch.tensor([[0, 1, 19, 3, 4], [10, 22, 9, 2, 7]], device="cuda")
    freq = 10000.0 ** (-torch.arange(0, 64, 2, device="cuda").float() / 64)
    angles = torch.arange(32, device="cuda")[:, None] * freq
    cache = torch.cat((angles.cos(), angles.sin()), -1)
    compare_gradients(
        Rotary.apply(x, positions, cache, 64, inverse),
        reference.rotary(ref, positions, 64, 10000.0, inverse),
        [x],
        [ref],
    )


def test_grouped_gemm_all_gradients_empty_expert():
    import deep_gemm

    alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    x = torch.randn(
        3 * alignment, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    w = torch.randn(
        3, 256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    rx, rw = clone(x), clone(w)
    ends = torch.tensor(
        [alignment, alignment, 3 * alignment], device="cuda", dtype=torch.int32
    )
    expected = torch.cat(
        [
            F.linear(rx[:alignment], rw[0]),
            F.linear(rx[alignment:alignment], rw[1]),
            F.linear(rx[alignment:], rw[2]),
        ]
    )
    compare_gradients(
        GroupedLinear.apply(x, w, ends, [alignment, 0, 2 * alignment]),
        expected,
        [x, w],
        [rx, rw],
        0.15,
    )
    assert torch.count_nonzero(w.grad[1]) == 0


def test_moe_fused_dispatch_padding_and_router_gradients():
    c = tiny_config()
    model = MoE(c).cuda().bfloat16()
    oracle = reference.MoE(c).cuda().bfloat16()
    with torch.no_grad():
        model.gate_up.normal_(std=0.02)
        model.down.normal_(std=0.02)
        oracle.gate.load_state_dict(model.gate.state_dict())
        oracle.shared_experts.load_state_dict(model.shared_experts.state_dict())
        for i, expert in enumerate(oracle.experts):
            expert.w1.weight.copy_(model.gate_up[i, :128])
            expert.w3.weight.copy_(model.gate_up[i, 128:])
            expert.w2.weight.copy_(model.down[i])
    x = torch.randn(2, 5, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    rx = clone(x)
    compare_gradients(
        model(x), oracle(rx), [x, model.gate.weight], [rx, oracle.gate.weight]
    )
    for i, expert in enumerate(oracle.experts):
        torch.testing.assert_close(
            model.gate_up.grad[i, :128], expert.w1.weight.grad, atol=0.04, rtol=0.04
        )
        torch.testing.assert_close(
            model.gate_up.grad[i, 128:], expert.w3.weight.grad, atol=0.04, rtol=0.04
        )
        torch.testing.assert_close(
            model.down.grad[i], expert.w2.weight.grad, atol=0.04, rtol=0.04
        )


def test_attention_private_windows_forward_and_backward():
    from specforge.modeling.draft.deepseek_v41.model import Attention

    c = tiny_config()
    layer = Attention(c).cuda().bfloat16()
    layer.sink.float()
    x = torch.randn(2, 3, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    context = torch.randn(
        2, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    rx, rc = clone(x), clone(context)
    positions = torch.tensor([[2, 3, 4], [9, 10, 11]], device="cuda")
    cp = torch.tensor([[0, 0, 0, 1], [5, 6, 7, 8]], device="cuda")
    valid = torch.tensor([[False, False, True, True], [True] * 4], device="cuda")
    freq = 10000.0 ** (-torch.arange(0, 64, 2, device="cuda").float() / 64)
    angles = torch.arange(12, device="cuda")[:, None] * freq
    cache = torch.cat((angles.cos(), angles.sin()), -1)
    y = layer(x, context, positions, cp, valid, cache)
    params = {name: clone(p) for name, p in layer.named_parameters()}

    def linear(a, name):
        return F.linear(a, params[name + ".weight"])

    def norm(a, name):
        return (
            a.float()
            * torch.rsqrt(a.float().square().mean(-1, keepdim=True) + c.rms_norm_eps)
            * params[name + ".weight"].float()
        ).to(a.dtype)

    q = linear(norm(linear(rx, "wq_a"), "q_norm"), "wq_b").unflatten(-1, (64, 512))
    q = reference.rotary(q, positions, 64, 10000.0)
    kv = norm(linear(torch.cat((rc, rx), 1), "wkv"), "kv_norm")
    kv = reference.rotary(kv, torch.cat((cp, positions), 1), 64, 10000.0)
    scores = torch.einsum("bkhd,bjd->bhkj", q.float(), kv.float()) / 512**0.5
    mask = torch.cat((valid, torch.ones_like(positions, dtype=torch.bool)), 1)
    scores = scores.masked_fill(~mask[:, None, None, :], -torch.inf)
    sink = params["sink.weight"][None, :, None, None].expand(*scores.shape[:-1], 1)
    prob = torch.cat((scores, sink), -1).softmax(-1)[..., :-1]
    out = torch.einsum("bhkj,bjd->bkhd", prob, kv.float()).bfloat16()
    out = reference.rotary(out, positions, 64, 10000.0, inverse=True).reshape(
        2, 3, 8, -1
    )
    projected = torch.einsum(
        "bkgd,grd->bkgr", out, params["wo_a.weight"].reshape(8, 64, -1)
    ).flatten(-2)
    expected = linear(projected, "wo_b")
    compare_gradients(
        y, expected, [x, context, *layer.parameters()], [rx, rc, *params.values()], 0.08
    )
