"""A zero confidence weight must statically disable the DSpark confidence head."""

import os
import unittest
from unittest import mock

import torch
import torch.distributed as dist
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.modeling.draft.dspark import DSparkDraftModel


def build_model(confidence_alpha):
    torch.manual_seed(3)
    config = Qwen3Config(
        architectures=["DSparkDraftModel"],
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        num_target_layers=2,
        head_dim=8,
        vocab_size=64,
        block_size=4,
        layer_types=["full_attention"],
        attn_implementation="sdpa",
        dflash_config={
            "projector_type": "dspark",
            "target_layer_ids": [0, 1],
            "mask_token_id": 63,
            "markov_head_type": "vanilla",
            "markov_rank": 8,
            "enable_confidence_head": True,
        },
    )
    return OnlineDSparkModel(
        DSparkDraftModel(config),
        nn.Linear(32, 64, bias=False).requires_grad_(False),
        nn.Embedding(64, 32).requires_grad_(False),
        mask_token_id=63,
        block_size=4,
        attention_backend="sdpa",
        num_anchors=4,
        dspark_confidence_head_alpha=confidence_alpha,
        fused_logits_loss=False,
    )


def batch():
    torch.manual_seed(7)
    return dict(
        input_ids=torch.randint(0, 62, (1, 16)),
        hidden_states=torch.randn(1, 16, 64),
        loss_mask=torch.ones(1, 16),
        target_last_hidden_states=torch.randn(1, 16, 32),
    )


class DSparkConfidenceFreezeTest(unittest.TestCase):
    def test_zero_weight_freezes_head_and_preserves_objective(self):
        enabled, disabled = build_model(1.0), build_model(0.0)
        head = disabled.draft_model.confidence_head
        self.assertFalse(any(p.requires_grad for p in head.parameters()))
        self.assertTrue(
            all(
                p.requires_grad
                for p in enabled.draft_model.confidence_head.parameters()
            )
        )
        # With the head's term removed, the objective is the same CE + L1.
        enabled.dspark_confidence_head_alpha = 0.0
        losses = []
        for model in (enabled, disabled):
            torch.manual_seed(9)
            loss, _, metrics = model(**batch())
            losses.append(loss)
            self.assertEqual(metrics["ratio_metrics"]["confidence_loss"][0].item(), 0)
        torch.testing.assert_close(losses[0], losses[1])

    def test_ddp_trains_without_unused_parameters(self):
        # Scoped: later tests treat a partial distributed environment as an error.
        environment = {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29541"}
        with mock.patch.dict(os.environ, environment):
            dist.init_process_group("gloo", rank=0, world_size=1)
        try:
            for alpha in (0.0, 1.0):
                with self.subTest(confidence_alpha=alpha):
                    ddp = nn.parallel.DistributedDataParallel(build_model(alpha))
                    for _ in range(2):
                        loss, _, _ = ddp(**batch())
                        loss.backward()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
