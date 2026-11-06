"""Compare whole-graph vs streaming loss through the real trainer and FSDP."""

import argparse
import gc
import json
import os
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from specforge.algorithms.deepseek_v41.strategy import NativeStrategy
from specforge.optimizer import BF16Optimizer
from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import DSparkTrainStrategy
from tests.test_modeling.test_dspark_anchor_padding import build_model


class DraftProjection(torch.nn.Module):
    """Callable block; loss heads that directly use weights stay in their owner."""

    def __init__(self, projection):
        super().__init__()
        self.projection = projection

    def forward(self, value):
        return self.projection(value)


def run_policy(streaming, rank, world_size, device):
    torch.manual_seed(42)
    model = build_model().to(device).bfloat16()
    model.anchor_sampling = "all"
    model.anchor_forward_chunk_blocks = 8
    # A weighted denominator must survive the root FSDP BF16 input cast.
    model.loss_decay_gamma = 2.3
    model.draft_model.projection = DraftProjection(model.draft_model.projection)
    model.draft_model._no_split_modules = ["DraftProjection"]
    model.draft_model.confidence_head.float()
    backend = FSDPTrainingBackend(
        ParallelConfig(
            world_size=world_size,
            sharding_strategy="FULL_SHARD",
            fsdp_process_group=dist.group.WORLD,
        ),
        optimizer_factory=lambda draft: BF16Optimizer(
            draft, lr=1e-4, total_steps=2, warmup_ratio=0, offload_master=True
        ),
    )
    backend.prepare_model(model, optimizer_target=model.draft_model)
    wrappers = [m for m in backend.module.modules() if isinstance(m, FSDP)]
    assert len(wrappers) == 3
    assert all(m.backward_prefetch is None and not m.forward_prefetch for m in wrappers)
    strategy = (
        NativeStrategy(backend.module)
        if streaming
        else DSparkTrainStrategy(backend.module)
    )
    core = TrainerCore(strategy, backend, accumulation_steps=4)
    initial = torch.cat(
        [p.detach().float().cpu().flatten() for p in backend.optimizer.fp32_params]
    )
    gradients, updates, rows = [], [], []
    step = backend.step

    def measured_step(**options):
        gradients.append(
            torch.cat(
                [
                    p.grad.detach().float().cpu().flatten()
                    for p in backend.optimizer.model_params
                    if p.grad is not None
                ]
            )
        )
        result = step(**options)
        updates.append(
            torch.cat(
                [
                    p.detach().float().cpu().flatten()
                    for p in backend.optimizer.fp32_params
                ]
            )
            - initial
        )
        return result

    backend.step = measured_step
    backward = backend.backward
    calls = [0]
    shard_sizes = [p.numel() for p in backend.optimizer.model_params]

    def measured_backward(loss, **options):
        backward(loss, **options)
        calls[0] += 1
        assert [p.numel() for p in backend.optimizer.model_params] == shard_sizes
        assert all(
            p.grad is None or p.grad.numel() == n
            for p, n in zip(backend.optimizer.model_params, shard_sizes)
        )
        assert all(
            m._handle is None
            or m._handle.flat_param._full_param_padded.untyped_storage().nbytes() == 0
            for m in wrappers
        )

    backend.backward = measured_backward
    for micro in range(8):
        valid = (0, 1, 7, 8, 19, 24, 25, 65)[(rank + micro) % 8]
        length = max(1, valid + 1)
        mask = torch.ones(1, length, device=device)
        if not valid:
            mask.zero_()
        batch = TrainBatch(
            sample_ids=[str(micro)],
            strategy="deepseek_v41",
            tensors={
                "input_ids": torch.randint(0, 30, (1, length), device=device),
                "loss_mask": mask,
                "hidden_states": torch.randn(
                    1, length, 16, device=device, dtype=torch.bfloat16
                ),
                "target_last_hidden_states": torch.randn(
                    1, length, 16, device=device, dtype=torch.bfloat16
                ),
            },
        )
        before = calls[0]
        result = core.train_step(batch)
        assert result.optimizer_stepped == ((micro + 1) % 4 == 0)
        counts = [None] * world_size
        dist.all_gather_object(counts, calls[0] - before)
        assert len(set(counts)) == 1
        metrics = result.materialize_metrics()
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        rows.append(
            {"micro": micro + 1, "backward_calls": counts[0], "metrics": metrics}
        )
    receipt = {"streaming": streaming, "optimizer_steps": len(updates), "cases": rows}
    del core, backend, model, strategy, wrappers
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return receipt, gradients, updates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    device = (
        torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        if args.device == "cuda"
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    else:
        torch.set_num_threads(1)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl" if device.type == "cuda" else "gloo",
        timeout=timedelta(minutes=10),
    )
    rank, world = dist.get_rank(), dist.get_world_size()
    args.output.mkdir(parents=True, exist_ok=True)
    original_init = FSDP.__init__

    def local_init(self, *a, **kw):
        if device.type == "cpu":
            kw["device_id"] = device
        return original_init(self, *a, **kw)

    try:
        with patch.object(FSDP, "__init__", local_init):
            reference, ref_grads, ref_updates = run_policy(False, rank, world, device)
            streamed, stream_grads, stream_updates = run_policy(
                True, rank, world, device
            )
        differences = []
        for label, left, right, tolerance in [
            ("gradient", ref_grads, stream_grads, 0.03),
            ("master_update", ref_updates, stream_updates, 0.10),
        ]:
            for step, (expected, actual) in enumerate(zip(left, right), 1):
                assert torch.isfinite(actual).all()
                relative = float(
                    (actual - expected).norm() / expected.norm().clamp_min(1e-12)
                )
                assert relative < tolerance, (label, step, relative)
                differences.append(
                    {
                        "kind": label,
                        "step": step,
                        "relative_l2": relative,
                        "tolerance": tolerance,
                    }
                )
        for a, b in zip(reference["cases"], streamed["cases"]):
            for key in a["metrics"]:
                if key.startswith("anchors/") or key.startswith("supervision/"):
                    assert a["metrics"][key] == b["metrics"][key], key
                elif key != "grad_norm":
                    strict = a["micro"] <= 4 and key == "loss"
                    torch.testing.assert_close(
                        torch.tensor(a["metrics"][key]),
                        torch.tensor(b["metrics"][key]),
                        rtol=2e-6 if strict else 0.02,
                        atol=2e-7 if strict else 0.002,
                        msg=key,
                    )
        receipt = {
            "passed": True,
            "rank": rank,
            "world_size": world,
            "device": str(device),
            "reference": reference,
            "streamed": streamed,
            "comparison": differences,
        }
        (args.output / f"rank{rank}.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
