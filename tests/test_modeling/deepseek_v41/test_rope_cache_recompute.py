"""Later anchor chunks must not change an earlier recomputation's RoPE shape."""

from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint

from specforge.modeling.draft.deepseek_v41.model import DeepSeekV41DraftModel


def test_cache_growth_between_forward_and_backward_preserves_values_and_gradients():
    state = SimpleNamespace(
        config=SimpleNamespace(rope_theta=10000, qk_rope_head_dim=64), rope_cache=None
    )
    lengths = (12, 13, 7, 30)
    inputs = [torch.randn(2, n, requires_grad=True) for n in lengths]
    references = []

    def forward(value):
        cache = DeepSeekV41DraftModel._rotary_cache(state, value.shape[1], value.device)
        return value @ cache

    outputs = []
    for value in inputs:
        outputs.append(checkpoint(forward, value, use_reentrant=False))
        references.append(forward(value).detach())
    assert state.rope_cache.shape == (30, 64)
    sum(output.sum() for output in outputs).backward()
    for value, output, reference in zip(inputs, outputs, references):
        torch.testing.assert_close(output.detach(), reference)
        expected = state.rope_cache[: value.shape[1]].sum(-1).expand_as(value)
        torch.testing.assert_close(value.grad, expected)
