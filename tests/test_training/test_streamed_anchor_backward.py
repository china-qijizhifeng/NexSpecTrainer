"""Streaming preserves the full microbatch objective and optimizer boundaries."""

import copy

import pytest
import torch

from specforge.algorithms.deepseek_v41.strategy import NativeStrategy
from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import DSparkTrainStrategy, StepContext
from tests.test_modeling.test_dspark_anchor_padding import build_model


class RecordingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.01)
        self.updates = 0

    def step(self):
        super().step()
        self.updates += 1
        self.zero_grad(set_to_none=True)


@pytest.mark.parametrize("chunk", [1, 7, 32])
@pytest.mark.parametrize("detailed", [False, True])
def test_streamed_objective_gradients_metrics_and_updates(chunk, detailed):
    torch.manual_seed(81)
    reference = build_model()
    reference.anchor_sampling = "all"
    reference.anchor_forward_chunk_blocks = chunk
    streamed = copy.deepcopy(reference)
    cores = []
    for model, strategy in (
        (reference, DSparkTrainStrategy),
        (streamed, NativeStrategy),
    ):
        backend = FSDPTrainingBackend(
            ParallelConfig(), optimizer_factory=lambda m: RecordingSGD(m.parameters())
        )
        backend.prepare_model(model, wrap=False, optimizer_target=model.draft_model)
        cores.append(TrainerCore(strategy(model), backend, accumulation_steps=2))
    for micro in range(4):
        mask = torch.ones(3, 20 + micro)
        mask[0, 3:8] = 0
        mask[1, 2:] = 0
        mask[2] = 0
        batch = TrainBatch(
            sample_ids=[str(micro)],
            strategy="deepseek_v41",
            tensors={
                "input_ids": torch.randint(0, 30, mask.shape),
                "loss_mask": mask,
                "hidden_states": torch.randn(*mask.shape, 16),
                "target_last_hidden_states": torch.randn(*mask.shape, 16),
            },
        )
        results = [
            c.train_step(batch, StepContext(collect_detailed_metrics=detailed))
            for c in cores
        ]
        torch.testing.assert_close(
            results[0]._metric_values, results[1]._metric_values, rtol=3e-5, atol=3e-6
        )
        for (name, a), (_, b) in zip(
            reference.named_parameters(), streamed.named_parameters()
        ):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6, msg=name)
            torch.testing.assert_close(a.grad, b.grad, rtol=3e-5, atol=3e-6, msg=name)
        for core, result in zip(cores, results):
            assert core.backend.optimizer.updates == (micro + 1) // 2
            assert result.optimizer_stepped == (micro % 2 == 1)


def test_stream_rejects_globally_empty_supervision_before_backward():
    model = build_model()
    model.anchor_sampling = "all"
    batch = TrainBatch(
        sample_ids=["empty"],
        strategy="deepseek_v41",
        tensors={
            "input_ids": torch.ones(1, 5, dtype=torch.long),
            "loss_mask": torch.zeros(1, 5),
            "hidden_states": torch.zeros(1, 5, 16),
            "target_last_hidden_states": torch.zeros(1, 5, 16),
        },
    )

    def unexpected_backward(_):
        raise AssertionError("empty supervision reached backward")

    with pytest.raises(RuntimeError, match="no supervised"):
        NativeStrategy(model).stream_backward(batch, unexpected_backward)


def test_chunk_gradients_keep_small_contributions_in_bfloat16():
    model = torch.nn.Linear(1, 1, bias=False).bfloat16()
    backend = FSDPTrainingBackend(ParallelConfig())
    backend.prepare_model(model, wrap=False)
    model.weight.grad = torch.ones_like(model.weight)
    # At a running sum of 1, BF16 cannot represent this increment. It must
    # survive hundreds of chunk backwards before the microbatch is rounded.
    for _ in range(256):
        backend.backward((model.weight * (1 / 1024)).sum())
    torch.testing.assert_close(model.weight.grad.float(), torch.tensor([[1.0]]))
    with backend.chunked_backward_context() as backward:
        for _ in range(256):
            backward((model.weight * (1 / 1024)).sum())
    torch.testing.assert_close(model.weight.grad.float(), torch.tensor([[1.25]]))


def test_failed_chunk_backward_does_not_publish_partial_gradients():
    model = torch.nn.Linear(1, 1, bias=False).bfloat16()
    backend = FSDPTrainingBackend(ParallelConfig())
    backend.prepare_model(model, wrap=False)
    with pytest.raises(RuntimeError, match="failed chunk"):
        with backend.chunked_backward_context() as backward:
            backward(model.weight.sum())
            raise RuntimeError("failed chunk")
    assert model.weight.grad is None
