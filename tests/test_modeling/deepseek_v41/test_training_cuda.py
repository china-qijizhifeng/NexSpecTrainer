"""Exercise native draft recomputation and all three DSpark objectives together."""

import pytest
import torch
from torch import nn

from specforge.algorithms.deepseek_v41.model import OnlineDeepSeekV41Model
from specforge.modeling.draft.deepseek_v41 import DeepSeekV41DraftModel
from specforge.modeling.draft.deepseek_v41.ops import MoE

from .test_checkpoint import tiny_config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Hopper GPU acceptance suite"
)


def test_native_dspark_optimizer_step_with_recomputation():
    from specforge.modeling.draft.deepseek_v41.validation import preflight

    preflight()
    torch.manual_seed(17)
    config = tiny_config()
    draft = DeepSeekV41DraftModel(config).cuda().bfloat16()
    for module in draft.modules():
        if getattr(module, "_specforge_keep_fp32", False):
            module.float()
        if isinstance(module, MoE):
            with torch.no_grad():
                module.gate_up.normal_(std=0.02)
                module.down.normal_(std=0.02)
    head = (
        nn.Linear(config.hidden_size, config.vocab_size, bias=False).cuda().bfloat16()
    )
    embedding = nn.Embedding(config.vocab_size, config.hidden_size).cuda().bfloat16()
    head.requires_grad_(False)
    embedding.requires_grad_(False)
    model = OnlineDeepSeekV41Model(
        draft,
        head,
        embedding,
        config.mask_token_id,
        block_size=config.block_size,
        attention_backend="megatron_csa",
        num_anchors=2,
        objective_chunk_blocks=1,
        objective_checkpointing=True,
    ).train()
    optimizer = torch.optim.AdamW(draft.parameters(), lr=1e-3)
    inputs = torch.randint(0, config.vocab_size - 1, (1, 12), device="cuda")
    mask = torch.ones_like(inputs, dtype=torch.float32)
    mask[:, :2] = 0
    features = torch.randn(
        1,
        12,
        config.hidden_size * len(config.target_layer_ids),
        device="cuda",
        dtype=torch.bfloat16,
    )
    final_states = torch.randn(
        1,
        12,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    watched = [draft.mtp[0].ffn.gate_up, draft.confidence_head.proj.weight]
    before = [parameter.detach().clone() for parameter in watched]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, accuracy, _ = model(inputs, features, mask, final_states)
        assert torch.isfinite(loss).all() and torch.isfinite(accuracy).all()
        loss.backward()
        for name, parameter in draft.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
        optimizer.step()
    for parameter, original in zip(watched, before):
        assert torch.count_nonzero(parameter.detach() - original) > 0
    assert head.weight.grad is None and embedding.weight.grad is None
