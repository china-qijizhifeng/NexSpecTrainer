"""Exhaustive, recomputed anchors must match an unchunked full objective."""

import copy

import pytest
import torch

from specforge.algorithms.common.anchor_coverage import all_anchor_positions
from tests.test_modeling.test_dspark_anchor_padding import build_model


@pytest.mark.parametrize("chunk", [1, 7, 32])
def test_all_anchor_loss_and_gradients_match_full_reference(chunk):
    torch.manual_seed(41)
    reference = build_model()
    reference.num_anchors = 65
    model = copy.deepcopy(reference)
    model.anchor_sampling = "all"
    model.anchor_forward_chunk_blocks = chunk
    mask = torch.ones(3, 66)
    mask[0, 6:13] = 0
    mask[0, 50:] = 0
    mask[1, 2:] = 0
    mask[2] = 0  # a rank/row with no candidates must keep the graph alive
    batch = dict(
        input_ids=torch.randint(0, 30, mask.shape),
        loss_mask=mask,
        hidden_states=torch.randn(3, 66, 16),
        target_last_hidden_states=torch.randn(3, 66, 16),
    )
    outputs = []
    for current in (reference, model):
        loss, accuracy, metrics = current(**batch)
        loss.backward()
        outputs.append(
            (
                loss.detach(),
                accuracy,
                metrics["ratio_metrics"],
                {
                    name: p.grad
                    for name, p in current.named_parameters()
                    if p.requires_grad
                },
            )
        )
    torch.testing.assert_close(outputs[0], outputs[1], rtol=3e-5, atol=3e-6)
    valid = (mask[:, :-1] > 0.5) & (mask[:, 1:] > 0.5)
    assert metrics["sum_metrics"]["anchors/selected"] == valid.sum()
    assert metrics["sum_metrics"]["anchors/duplicates"] == 0


def test_every_legal_anchor_once_and_no_crossing_masked_span():
    model = build_model()
    mask = torch.tensor([[0, 1, 1, 0, 1, 1, 1, 1, 0, 1]], dtype=torch.float)
    positions = all_anchor_positions(mask)
    assert positions.tolist() == [[1, 4, 5, 6]]
    _, targets, _ = model._build_dspark_labels_and_mask(
        torch.arange(10).unsqueeze(0),
        mask,
        positions,
        positions >= 0,
    )
    assert targets.sum(-1).tolist() == [[1, 3, 2, 1]]
    # First call token (position 4) cannot be predicted across a masked anchor;
    # its remaining tokens are covered. We preserve the anchor legality rule.
    assert not targets[0, 0, 2]


def test_empty_window_has_only_masked_padding():
    assert all_anchor_positions(torch.zeros(2, 1)).tolist() == [[-1], [-1]]
