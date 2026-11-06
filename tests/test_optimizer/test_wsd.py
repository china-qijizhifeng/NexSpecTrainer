"""WSD config-to-update behavior and checkpoint continuation on CPU."""

import io

import pytest
import torch

from specforge.config import Config, TrainingConfig, apply_overrides
from specforge.lr_scheduler import WarmupStableDecayLR
from specforge.training.assembly import _ConfiguredOptimizerFactory

WSD = dict(
    lr_scheduler="wsd",
    total_steps=8,
    warmup_steps=2,
    stable_steps=2,
    warmup_lr=0.002,
    stable_lr=0.01,
)


def build(algorithm="adamw", **overrides):
    training = dict(WSD, optimizer=algorithm, fsdp_sharding="NO_SHARD")
    if algorithm == "muon_adamw":
        training["muon"] = dict(muon_lr_multiplier=2, adamw_lr_multiplier=0.5)
    training.update(overrides)
    cfg = Config.model_validate(
        {
            "model": {
                "target_model_path": "unused/target",
                "draft_model_config": "unused.json",
                "vocab_mapping_path": "unused.pt",
            },
            "data": {"hidden_states_path": "/unused"},
            "training": training,
        }
    )
    model = torch.nn.Linear(3, 2)
    factory = _ConfiguredOptimizerFactory(cfg)
    factory.prepare_model(model)
    return model, factory(model, wrapped_model=model)


def update(model, optimizer):
    # A real forward/backward/update, sensitive to LR and restored moments.
    model(torch.tensor([[0.1, -0.2, 0.3]])).square().mean().backward()
    optimizer.step()


@pytest.mark.parametrize("algorithm", ["adamw", "muon_adamw"])
def test_wsd_rates_used_by_actual_optimizer_updates(algorithm):
    model, optimizer = build(algorithm)
    expected = [
        0.002,
        0.006,
        0.01,
        0.01,
        0.01,
        0.008535533905932738,
        0.005,
        0.0014644660940672626,
        0.0,
        0.0,
    ]
    multipliers = [1] if algorithm == "adamw" else [2, 0.5]
    for rate in expected:
        assert [g["lr"] for g in optimizer.optimizer.param_groups] == pytest.approx(
            [rate * multiplier for multiplier in multipliers]
        )
        update(model, optimizer)


@pytest.mark.parametrize("algorithm", ["adamw", "muon_adamw"])
@pytest.mark.parametrize("split", [1, 2, 4, 5, 8])
def test_wsd_checkpoint_matches_uninterrupted_training(algorithm, split, monkeypatch):
    # The rank-zero logger assumes a distributed process group. Keep CPU
    # serialization, restore, and parameter updates real; only silence logging.
    monkeypatch.setattr("specforge.optimizer.print_on_rank0", lambda *args: None)
    model, optimizer = build(algorithm)
    for _ in range(split):
        update(model, optimizer)
    checkpoint = io.BytesIO()
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint
    )
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    resumed_model, resumed = build(algorithm)
    resumed_model.load_state_dict(saved["model"])
    resumed.load_state_dict(saved["optimizer"])
    for _ in range(split, 10):
        assert resumed.scheduler.get_last_lr() == optimizer.scheduler.get_last_lr()
        assert [g["lr"] for g in resumed.optimizer.param_groups] == [
            g["lr"] for g in optimizer.optimizer.param_groups
        ]
        update(model, optimizer)
        update(resumed_model, resumed)
        for expected, actual in zip(model.parameters(), resumed_model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "changes",
    [
        dict(warmup_steps=1),
        dict(stable_steps=1),
        dict(warmup_lr=0.001),
        dict(stable_lr=0.02),
        dict(total_steps=9),
    ],
)
def test_wsd_resume_rejects_changed_schedule_before_loading_optimizer(changes):
    model, optimizer = build()
    update(model, optimizer)
    _, resumed = build(**changes)
    original_rate = resumed.get_learning_rate()
    with pytest.raises(ValueError, match="checkpoint WSD scheduler"):
        resumed.load_state_dict(optimizer.state_dict())
    assert resumed.get_learning_rate() == original_rate
    assert not resumed.optimizer.state


@pytest.mark.parametrize(
    "warmup,stable,expected",
    [
        (0, 0, [0.01, 0.005, 0]),
        (0, 1, [0.01, 0.01, 0]),
        (1, 0, [0.002, 0.01, 0]),
    ],
)
def test_wsd_empty_phases_and_single_decay_step(warmup, stable, expected):
    model, optimizer = build(total_steps=2, warmup_steps=warmup, stable_steps=stable)
    for lr in expected:
        assert optimizer.get_learning_rate() == pytest.approx(lr)
        update(model, optimizer)


@pytest.mark.parametrize(
    "changes",
    [
        dict(warmup_steps=None),
        dict(stable_steps=None),
        dict(warmup_lr=None),
        dict(stable_lr=None),
        dict(warmup_steps=-1),
        dict(stable_steps=-1),
        dict(warmup_steps=1.5),
        dict(stable_steps=True),
        dict(warmup_lr=-0.1),
        dict(warmup_lr=0.02),
        dict(stable_lr=0),
        dict(stable_lr=float("inf")),
        dict(warmup_lr=float("nan")),
        dict(stable_steps=6),
        dict(stable_steps=8),
    ],
)
def test_wsd_invalid_config_and_direct_scheduler(changes):
    options = {**WSD, **changes}
    with pytest.raises(ValueError):
        TrainingConfig(**options)
    options.pop("lr_scheduler")
    inner = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.01)
    with pytest.raises(ValueError):
        WarmupStableDecayLR(inner, **options)


def test_wsd_dotted_overrides_and_deferred_horizon():
    cfg = Config.model_validate(
        {
            "model": {
                "target_model_path": "unused",
                "draft_model_config": "unused.json",
            },
            "data": {"hidden_states_path": "/unused"},
        }
    )
    cfg = apply_overrides(
        cfg,
        [
            "training.lr_scheduler=wsd",
            "training.warmup_steps=2",
            "training.stable_steps=2",
            "training.warmup_lr=0",
            "training.stable_lr=1e-4",
        ],
    )
    factory = _ConfiguredOptimizerFactory(cfg)
    factory.configure_total_steps(4)
    with pytest.raises(ValueError, match="warmup_steps.*stable_steps.*total_steps"):
        factory(torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match="require lr_scheduler=wsd"):
        TrainingConfig(warmup_steps=2)
