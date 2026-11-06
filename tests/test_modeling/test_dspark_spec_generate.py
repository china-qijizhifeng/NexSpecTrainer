"""DSpark spec_generate must propose from every slot, as training and SGLang do."""

import unittest
from unittest import mock

import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel

VOCAB = 64
BLOCK = 4
MASK = torch.tensor([0])


def draft_config(architecture, **dflash_config):
    return Qwen3Config(
        architectures=[architecture],
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        num_target_layers=2,
        head_dim=8,
        vocab_size=VOCAB,
        block_size=BLOCK,
        layer_types=["full_attention"],
        attn_implementation="sdpa",
        dflash_config={"target_layer_ids": [0, 1], "mask_token_id": 0, **dflash_config},
    )


def build_target():
    torch.manual_seed(5)
    config = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        head_dim=8,
        max_position_embeddings=128,
        vocab_size=VOCAB,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    target = Qwen3ForCausalLM(config).eval()
    # spec_generate drops mask-token ids from its output; never generate one.
    target.lm_head.register_forward_hook(
        lambda _module, _inputs, logits: logits.index_fill(-1, MASK, -1e9)
    )
    return target


@torch.no_grad()
def greedy(target, input_ids, steps):
    ids = input_ids
    for _ in range(steps):
        ids = torch.cat([ids, target(ids).logits[:, -1:].argmax(-1)], dim=1)
    return ids


class OracleTarget(nn.Module):
    """The real target, except its LM head maps a slot's position to the
    greedy token one position later: exactly what DSpark training supervises."""

    def __init__(self, target, sequence):
        super().__init__()
        self.inner, self.model, self.sequence = target, target.model, sequence
        self.verify_lengths = []
        self.lm_head = self._oracle_head

    @property
    def device(self):
        return self.inner.device

    def _oracle_head(self, hidden):
        next_ids = self.sequence[0, hidden[..., 0].long() + 1]
        return torch.nn.functional.one_hot(next_ids, VOCAB).float() * 1e4

    def forward(self, input_ids, **kwargs):
        if kwargs.get("logits_to_keep") != 1:
            self.verify_lengths.append(input_ids.shape[1])
        return self.inner(input_ids, **kwargs)


class DSparkSpecGenerateTest(unittest.TestCase):
    def test_every_slot_proposes_the_token_after_its_position(self):
        model = DSparkDraftModel(
            draft_config(
                "DSparkDraftModel",
                projector_type="dspark",
                markov_head_type="vanilla",
                markov_rank=4,
            )
        ).eval()
        torch.manual_seed(2)
        prompt = torch.randint(1, VOCAB, (1, 6))
        new_tokens = 2 * (BLOCK + 1)
        sequence = greedy(build_target(), prompt, new_tokens + BLOCK + 2)
        target = OracleTarget(build_target(), sequence)

        def oracle_draft(*, noise_embedding, position_ids, **_):
            hidden = torch.zeros(1, noise_embedding.shape[1], 32)
            hidden[..., 0] = position_ids[:, -noise_embedding.shape[1] :]
            return hidden

        with mock.patch.object(model, "forward", side_effect=oracle_draft):
            output = model.spec_generate(
                target,
                prompt,
                max_new_tokens=new_tokens,
                stop_token_ids=None,
                temperature=0.0,
            )
        torch.testing.assert_close(output, sequence[:, : output.shape[1]])
        self.assertEqual(output.shape[1], prompt.shape[1] + new_tokens)
        # Anchor plus BLOCK proposals per verify; all accepted, plus the bonus.
        self.assertEqual(target.verify_lengths, [BLOCK + 1, BLOCK + 1])

    def test_markov_chain_starts_from_the_anchor(self):
        model = DSparkDraftModel(
            draft_config(
                "DSparkDraftModel",
                projector_type="dspark",
                markov_head_type="vanilla",
                markov_rank=4,
            )
        ).eval()
        head = nn.Linear(32, VOCAB, bias=False)
        hidden = torch.randn(1, 3 + BLOCK, 32)
        block = torch.tensor([[7] + [0] * (BLOCK - 1)])
        with mock.patch.object(
            model.markov_head,
            "sample_block_tokens",
            wraps=model.markov_head.sample_block_tokens,
        ) as sample:
            tokens = model._sample_draft_tokens(mock.Mock(lm_head=head), hidden, block)
        self.assertEqual(tokens.shape, (1, BLOCK))
        kwargs = sample.call_args.kwargs
        torch.testing.assert_close(kwargs["first_prev_token_ids"], torch.tensor([7]))
        torch.testing.assert_close(kwargs["hidden_states"], hidden[:, -BLOCK:])

    def test_greedy_output_matches_target_for_dflash_and_dspark(self):
        torch.manual_seed(2)
        prompt = torch.randint(1, VOCAB, (1, 6))
        target = build_target()
        expected = greedy(target, prompt, 10)
        for model in (
            DFlashDraftModel(draft_config("DFlashDraftModel")),
            DSparkDraftModel(
                draft_config(
                    "DSparkDraftModel",
                    projector_type="dspark",
                    markov_head_type="rnn",
                    markov_rank=4,
                )
            ),
        ):
            with self.subTest(draft=type(model).__name__):
                torch.manual_seed(11)
                output = model.eval().spec_generate(
                    target,
                    prompt,
                    max_new_tokens=10,
                    stop_token_ids=None,
                    temperature=0.0,
                )
                torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    unittest.main()
