"""Expert chunking must keep every token reduction intact, including empty groups."""

import sys
from types import SimpleNamespace

import pytest
import torch

from specforge.modeling.draft.deepseek_v41.kernels.moe import grouped_weight_gradient


@pytest.mark.parametrize("chunk", [1, 3, 8])
def test_expert_chunks_match_independent_full_token_reductions(monkeypatch, chunk):
    counts = [0] * 8 + [3, 0, 7, 2, 0, 1, 5, 4, 0]
    torch.manual_seed(25)
    x = torch.randn(sum(counts), 6).bfloat16()
    dy = torch.randn(sum(counts), 5).bfloat16()
    weight = torch.empty(len(counts), 5, 6, dtype=torch.bfloat16)
    observed = []

    def kernel(left, right, out, lengths, gpu_lengths, addend):
        assert lengths == gpu_lengths.tolist()
        assert len(lengths) <= chunk
        assert sum(lengths) == left.shape[0] == right.shape[0]
        assert out.data_ptr() == addend.data_ptr()
        observed.append(out.numel())
        row = 0
        for i, n in enumerate(lengths):
            out[i].add_(left[row : row + n].float().T @ right[row : row + n].float())
            row += n

    monkeypatch.setitem(
        sys.modules,
        "deep_gemm",
        SimpleNamespace(k_grouped_bf16_gemm_tn_contiguous=kernel),
    )
    actual = grouped_weight_gradient(dy, x, weight, counts, expert_chunk=chunk)
    expected = []
    row = 0
    for n in counts:
        expected.append(
            (dy[row : row + n].float().T @ x[row : row + n].float()).bfloat16()
        )
        row += n
    torch.testing.assert_close(actual, torch.stack(expected), rtol=0, atol=0)
    assert max(observed) <= chunk * 5 * 6
