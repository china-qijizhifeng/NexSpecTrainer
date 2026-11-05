"""Overlapping anchor windows must not accumulate their context grads in BF16."""

import torch

from specforge.modeling.draft.deepseek_v41.model import gather_context_windows


def test_overlapping_context_gradient_matches_fp32_accumulation():
    torch.manual_seed(42)
    context = torch.randn(1, 256, 8, dtype=torch.bfloat16, requires_grad=True)
    reference = context.detach().float().requires_grad_()
    indices = torch.arange(128)[:, None] + torch.arange(128)[None, :]
    rows = torch.zeros(128, 1, dtype=torch.long)
    weights = torch.rand(128, 128, 8, dtype=torch.bfloat16) * 0.01
    actual = gather_context_windows(context, rows, indices)
    expected = reference[rows, indices]
    torch.testing.assert_close(actual, expected.bfloat16(), rtol=0, atol=0)
    (actual * weights).float().sum().backward()
    (expected * weights.float()).sum().backward()
    torch.testing.assert_close(context.grad, reference.grad.bfloat16(), rtol=0, atol=0)

    # Negative control: the previous native call site rounds at each add.
    context.grad = None
    (context[rows, indices] * weights).float().sum().backward()
    assert not torch.equal(context.grad, reference.grad.bfloat16())
