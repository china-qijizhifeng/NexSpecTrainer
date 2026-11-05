"""Masked anchor slots must preserve loss/gradients and FSDP call counts."""

import copy
import unittest
from types import MethodType, SimpleNamespace

import torch
from torch import nn

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.algorithms.deepseek_v41.model import OnlineDeepSeekV41Model
from specforge.modeling.draft.deepseek_v41.model import ConfidenceHead


class TinyDraft(nn.Module):
    """Independent draft blocks with the production native confidence head."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(16, 16, bias=False)
        self.markov = nn.Embedding(32, 8)
        self.confidence_head = ConfidenceHead(
            SimpleNamespace(hidden_size=16, markov_rank=8)
        )

    def forward(self, *, noise_embedding, **kwargs):
        return self.projection(noise_embedding)

    def normalize_for_logits(self, hidden):
        return hidden

    def apply_logits_head(self, logits, **kwargs):
        return logits

    def predict_confidence(self, hidden, *, prev_token_ids):
        return self.confidence_head(hidden, self.markov(prev_token_ids))


def build_model():
    return OnlineDeepSeekV41Model(
        TinyDraft(),
        nn.Linear(16, 32, bias=False).requires_grad_(False),
        nn.Embedding(32, 16).requires_grad_(False),
        mask_token_id=31,
        block_size=5,
        num_anchors=32,
        objective_chunk_blocks=8,
        attention_backend="megatron_csa",
        fused_logits_loss=False,
    )


class DSparkAnchorPaddingTest(unittest.TestCase):
    def test_fixed_width_preserves_selected_anchors_and_masks_short_sequences(self):
        model = build_model()
        for valid in (1, 7, 8, 9, 19, 24, 25, 32, 48):
            mask = torch.ones(1, valid + 1)
            torch.manual_seed(4)
            reference, reference_keep = OnlineDFlashModel._sample_anchor_positions(
                model, mask.shape[1], mask, mask.device, max_valid_anchors=valid
            )
            torch.manual_seed(4)
            anchors, keep = model._sample_anchor_positions(
                mask.shape[1], mask, mask.device, max_valid_anchors=valid
            )
            self.assertEqual(anchors.shape, (1, 32))
            self.assertEqual(keep.sum().item(), min(32, valid))
            torch.testing.assert_close(anchors[keep], reference[reference_keep])
            self.assertTrue((anchors[~keep] == 0).all())
        with self.assertRaisesRegex(ValueError, "consecutive supervised"):
            model._sample_anchor_positions(2, torch.zeros(1, 2), torch.device("cpu"))

    def test_padding_preserves_objective_and_all_parameter_gradients(self):
        torch.manual_seed(17)
        model = build_model()
        reference = copy.deepcopy(model)
        reference._sample_anchor_positions = MethodType(
            OnlineDFlashModel._sample_anchor_positions, reference
        )
        for valid in (1, 8, 19, 25, 48):
            for checkpointing in (False, True):
                # Rows with unequal supervision also exercise existing batch padding.
                mask = torch.ones(2, valid + 1)
                mask[1, min(3, valid + 1) :] = 0
                batch = dict(
                    input_ids=torch.randint(0, 30, mask.shape),
                    loss_mask=mask,
                    hidden_states=torch.randn(2, valid + 1, 16),
                    target_last_hidden_states=torch.randn(2, valid + 1, 16),
                )
                outputs = []
                for current in (reference, model):
                    current.zero_grad(set_to_none=True)
                    current.objective_checkpointing = checkpointing
                    torch.manual_seed(91)
                    loss, accuracy, metrics = current(**batch)
                    loss.backward()
                    gradients = {
                        name: p.grad.clone()
                        for name, p in current.named_parameters()
                        if p.requires_grad
                    }
                    self.assertTrue(
                        all(torch.isfinite(g).all() for g in gradients.values())
                    )
                    outputs.append((loss.detach(), accuracy, metrics, gradients))
                torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
