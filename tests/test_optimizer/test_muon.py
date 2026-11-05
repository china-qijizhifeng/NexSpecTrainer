"""Numerical and restore contracts; accelerator/FSDP coverage lives in tests/gpu."""

import copy

import pytest
import torch
from torch import nn

from specforge.config.schema import TrainingConfig
from specforge.muon import MuonAdamWOptimizer, muon_direction
from specforge.muon_config import MuonConfig
from specforge.muon_layout import capture_layout
from specforge.optimizer import BF16Optimizer


def build(model, config=None, **kwargs):
    config = config or MuonConfig()
    return MuonAdamWOptimizer(
        model,
        plan=capture_layout(model, config),
        config=config,
        lr=0.003,
        total_steps=20,
        warmup_ratio=0,
        lr_scheduler="constant",
        max_grad_norm=1e6,
        **kwargs,
    )


@pytest.mark.parametrize("shape", [(5, 13), (13, 5), (7, 7)])
@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("scaling", ["original", "match_rms_adamw"])
def test_multistep_against_pytorch_muon_and_adamw(shape, nesterov, scaling):
    if not hasattr(torch.optim, "Muon"):
        pytest.skip(
            "PyTorch Muon is not installed; pinned TorchTitan tested separately"
        )
    torch.manual_seed(121)
    model = nn.Linear(shape[1], shape[0])
    reference = copy.deepcopy(model)
    config = MuonConfig(nesterov=nesterov, scaling=scaling)
    optimizer = build(model, config)
    muon = torch.optim.Muon(
        [reference.weight],
        lr=0.003,
        weight_decay=0.1,
        nesterov=nesterov,
        adjust_lr_fn=scaling,
    )
    adam = torch.optim.AdamW([reference.bias], lr=0.003, weight_decay=0)
    for step in range(4):
        for p, q in zip(model.parameters(), reference.parameters()):
            p.grad = torch.zeros_like(p) if step == 2 else torch.randn_like(p)
            q.grad = p.grad.clone()
        optimizer.step()
        muon.step()
        adam.step()
        torch.testing.assert_close(model.weight, reference.weight, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(model.bias, reference.bias, atol=1e-7, rtol=1e-6)


def test_v41_published_algorithm_has_ten_iterations_and_spectral_rescaling():
    config = MuonConfig(profile="v41_flash_muon", ns_dtype="float32")
    assert (
        config.ns_coefficients
        == ((3.4445, -4.775, 2.0315),) * 8 + ((2.0, -1.5, 0.5),) * 2
    )
    assert config.momentum_convention == "sum"
    assert config.adamw_betas == (0.9, 0.95) and config.adamw_eps == 1e-20
    # Identity is a fixed point after normalization and converges to itself.
    direction = muon_direction(torch.eye(4), config)
    scalar = 0.5
    for a, b, c in config.ns_coefficients:
        scalar = a * scalar + b * scalar**3 + c * scalar**5
    torch.testing.assert_close(
        direction, torch.eye(4) * (scalar * 0.36), atol=2e-6, rtol=2e-6
    )
    assert torch.count_nonzero(muon_direction(torch.zeros(4, 7), config)) == 0


@pytest.mark.parametrize(
    "mutation",
    [
        dict(momentum=0.9),
        dict(ns_steps=5),
        dict(nesterov=False),
        dict(scaling="rms"),
        dict(adamw_eps=1e-8),
    ],
)
def test_v41_name_cannot_silently_select_a_different_algorithm(mutation):
    with pytest.raises(ValueError, match="v41_flash_muon|schedule"):
        MuonConfig(profile="v41_flash_muon", **mutation)


@pytest.mark.parametrize(
    "mutation",
    [
        dict(momentum=float("nan")),
        dict(adamw_betas=(0.9, 1)),
        dict(ns_steps=0),
        dict(ns_coefficients=()),
        dict(unknown=True),
    ],
)
def test_invalid_config(mutation):
    with pytest.raises(ValueError):
        MuonConfig(**mutation)


def test_training_config_resolves_named_algorithm_and_rejects_unsupported_sharding():
    t = TrainingConfig(optimizer="v41_flash_muon", fsdp_sharding="FULL_SHARD")
    assert t.muon == MuonConfig(profile="v41_flash_muon")
    assert TrainingConfig().optimizer == "adamw"
    with pytest.raises(ValueError, match="FULL_SHARD"):
        TrainingConfig(optimizer="v41_flash_muon")
    with pytest.raises(ValueError):
        TrainingConfig(muon=MuonConfig())


@pytest.mark.parametrize("lazy", [True, False])
def test_resume_next_update_and_algorithm_rejection(lazy, monkeypatch):
    monkeypatch.setattr("specforge.optimizer.print_on_rank0", lambda *args: None)
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(11, 7), nn.LayerNorm(7))
    opt = build(model)
    if not lazy:
        model(torch.randn(2, 11)).square().sum().backward()
        opt.step()
    saved = copy.deepcopy(opt.state_dict())
    fresh_model = copy.deepcopy(model)
    fresh = build(fresh_model)
    fresh.load_state_dict(saved)
    for p, q in zip(model.parameters(), fresh_model.parameters()):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
    opt.step()
    fresh.step()
    for p, q in zip(opt.fp32_params, fresh.fp32_params):
        assert torch.equal(p, q)
    adam = BF16Optimizer(
        fresh_model,
        lr=0.003,
        total_steps=20,
        warmup_ratio=0,
        lr_scheduler="constant",
        max_grad_norm=1e6,
    )
    with pytest.raises(ValueError, match="weights-only"):
        adam.load_state_dict(saved)
    with pytest.raises(ValueError, match="weights-only"):
        fresh.load_state_dict(adam.state_dict())
    incompatible = build(copy.deepcopy(model), MuonConfig(momentum=0.8))
    with pytest.raises(ValueError, match="weights-only"):
        incompatible.load_state_dict(saved)


def test_nonfinite_gradient_does_not_partially_update_parameters_or_states():
    model = nn.Sequential(nn.Linear(11, 7), nn.LayerNorm(7))
    opt = build(model)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    before = copy.deepcopy(opt.state_dict())
    weights = [p.detach().clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    list(model.parameters())[-1].grad[0] = float("nan")
    with pytest.raises(FloatingPointError):
        opt.step()
    for p, q in zip(model.parameters(), weights):
        assert torch.equal(p, q)
    after = opt.state_dict()
    assert before["scheduler_state_dict"] == after["scheduler_state_dict"]
    for index, state in before["optimizer_state_dict"]["state"].items():
        for name, value in state.items():
            assert torch.equal(
                value, after["optimizer_state_dict"]["state"][index][name]
            )


def test_native_role_layout_uses_original_expert_and_head_shapes():
    from specforge.modeling.draft.deepseek_v41.config import DeepSeekV41DraftConfig
    from specforge.modeling.draft.deepseek_v41.model import DeepSeekV41DraftModel

    c = DeepSeekV41DraftConfig(
        hidden_size=16,
        vocab_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        o_groups=2,
        o_lora_rank=8,
        moe_intermediate_size=12,
        n_routed_experts=3,
        markov_rank=4,
        target_layer_ids=[0],
    )
    model = DeepSeekV41DraftModel(c)
    plan = {
        p.name: p for p in capture_layout(model, MuonConfig(profile="v41_flash_muon"))
    }
    assert plan["mtp.0.attn.wq_b.weight"].matrix_shape == (8, 8)
    assert plan["mtp.0.attn.wkv.weight"].matrix_shape == (8, 16)
    assert plan["mtp.0.ffn.gate_up"].shape == (3, 24, 16)
    assert plan["mtp.0.ffn.gate_up"].matrix_shape == (12, 16)
    assert plan["mtp.0.ffn.down"].matrix_shape == (16, 12)
    assert plan["mtp.0.hc_attn.fn"].matrix_shape is not None
    assert plan["mtp.0.ffn.gate.weight"].matrix_shape is not None
    assert plan["mtp.0.markov_head.embed.weight"].matrix_shape is None
    assert plan["mtp.0.markov_head.head.weight"].matrix_shape is None
    assert plan["mtp.0.attn_norm.weight"].role == "normalization"
    assert not any("bias_vl" in name or name.endswith("gate.bias") for name in plan)


def test_v41_multistep_matches_paper_scalar_oracle():
    """An independent double-precision oracle on diagonal logical matrices."""
    import math

    from specforge.muon_config import V41_FLASH_DEFAULTS

    config = MuonConfig(**V41_FLASH_DEFAULTS, ns_dtype="float32")
    model = nn.Linear(5, 3, bias=False)
    model.weight.data.zero_()
    optimizer = build(model, config)
    momentum = [0.0, 0.0, 0.0]
    expected = [0.0, 0.0, 0.0]
    for values in [
        [1.0, 2.0, 3.0],
        [-2.0, 0.3, 4.0],
        [0.0, 0.0, 0.0],
        [0.2, -0.3, 1.0],
    ]:
        model.weight.grad = torch.zeros_like(model.weight)
        for i, g in enumerate(values):
            model.weight.grad[i, i] = g
            momentum[i] = 0.95 * momentum[i] + g
        direction = [g + 0.95 * m for g, m in zip(values, momentum)]
        norm = math.sqrt(sum(x * x for x in direction))
        direction = [x / max(norm, 1e-7) for x in direction]
        for a, b, c in config.ns_coefficients:
            direction = [a * x + b * x**3 + c * x**5 for x in direction]
        expected = [
            w * (1 - 0.003 * 0.1) - 0.003 * 0.18 * math.sqrt(5) * x
            for w, x in zip(expected, direction)
        ]
        optimizer.step()
        torch.testing.assert_close(
            model.weight.diag(), torch.tensor(expected), atol=1e-7, rtol=1e-5
        )
        actual_momentum = optimizer.optimizer.state[optimizer.fp32_params[0]][
            "momentum_buffer"
        ]
        torch.testing.assert_close(actual_momentum.diag(), torch.tensor(momentum))


def test_empty_adam_shard_does_not_allocate_uncheckpointable_empty_state():
    model = nn.Linear(3, 2)
    config = MuonConfig()
    plan = capture_layout(model, config)
    # Simulate FSDP's zero-length original-parameter view on this rank.
    model.bias.data = torch.empty(0)
    opt = MuonAdamWOptimizer(
        model, plan=plan, config=config, lr=0.001, total_steps=3, warmup_ratio=0
    )
    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.empty_like(model.bias)
    opt.step()
    assert opt.fp32_params[1] not in opt.optimizer.state
    assert set(opt.optimizer.state_dict()["state"]) == {0}
