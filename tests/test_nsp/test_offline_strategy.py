import torch

from specforge.nsp.offline_eval import output_terms
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import DSparkTrainStrategy
from tests.test_modeling.test_dspark_anchor_padding import build_model


def test_dspark_evaluation_terms_do_not_replace_training_gradients():
    torch.manual_seed(17)
    model = build_model()
    strategy = DSparkTrainStrategy(model)
    batch = TrainBatch(
        ["sample"],
        "deepseek_v41",
        {
            "input_ids": torch.randint(0, 30, (1, 12)),
            "loss_mask": torch.ones(1, 12),
            "hidden_states": torch.randn(1, 12, 16),
            "target_last_hidden_states": torch.randn(1, 12, 16),
            "eval_anchor_positions": torch.tensor([[0, 4, 8]]),
        },
    )
    train = strategy.forward_loss(batch)
    assert train.loss_terms is None
    train.loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.draft_model.parameters()
    )
    with torch.no_grad():
        evaluated = strategy.forward_loss(batch)
    terms = output_terms(evaluated, ["loss", "accuracy", "prefix-al"], "deepseek_v41")
    assert abs(terms["loss"][0] / terms["loss"][1] - evaluated.loss.item()) < 1e-6
    assert terms["prefix-al"][1] == 2  # last anchor lacks a full five-token horizon
