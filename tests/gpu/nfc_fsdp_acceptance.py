"""Run with torchrun --nproc_per_node=8; real native NFC + nested FSDP1.

Artifacts go only to the explicit --output directory. No pretrained data needed.
"""

import argparse
import gc
import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from specforge.optimizer import BF16Optimizer
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.checkpoint import CheckpointManager
from specforge.training.controller import TrainerController
from specforge.training.nfc_checkpoint import load_draft_weights
from specforge.training.nfc_state import model_slices


class FP32Scale(torch.nn.Module):
    _specforge_keep_fp32 = True

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(29))

    def forward(self, x):
        return (x.float() * self.scale).to(x.dtype)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(29, 29)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        return x + self.dropout(torch.tanh(self.linear(x)))


class Draft(torch.nn.Module):
    _no_split_modules = ("Block",)

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(17, 29)
        self.blocks = torch.nn.Sequential(Block(), Block())
        self.fp32 = FP32Scale()
        self.norm = torch.nn.LayerNorm(29)
        self.register_buffer("indices", torch.arange(3))
        self.register_buffer("empty", torch.empty(0))

    def forward(self, x):
        return self.norm(self.fp32(self.blocks(self.proj(x))))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.draft_model = Draft().bfloat16()
        self.draft_model.fp32.float()
        self.lm_head = torch.nn.Linear(29, 17).bfloat16().requires_grad_(False)

    def forward(self, x):
        return self.lm_head(self.draft_model(x))


def state_filter(state):
    return {
        k.removeprefix("draft_model."): v
        for k, v in state.items()
        if k.startswith("draft_model.")
    }


def create_backend(offload, strategy):
    torch.manual_seed(42)
    model = Model().cuda()
    backend = FSDPTrainingBackend(
        ParallelConfig(
            world_size=dist.get_world_size(),
            sharding_strategy=strategy,
            fsdp_process_group=dist.group.WORLD,
        ),
        optimizer_factory=lambda draft: BF16Optimizer(
            draft, lr=1e-3, total_steps=20, warmup_ratio=0.1, offload_master=offload
        ),
    )
    backend.prepare_model(model, optimizer_target=model.draft_model)

    def no_full_gather():
        raise AssertionError("native checkpoint called FULL_STATE_DICT")

    backend._module_state_dict = no_full_gather
    return backend


def fingerprint(value):
    digest = hashlib.sha256()

    def visit(x):
        digest.update(type(x).__qualname__.encode())
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().contiguous()
            digest.update(str((str(t.dtype), tuple(t.shape))).encode())
            digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            digest.update(str((x.dtype, x.shape)).encode())
            digest.update(x.tobytes())
        elif isinstance(x, dict):
            for k, v in x.items():
                visit(k)
                visit(v)
        elif isinstance(x, (tuple, list)):
            for item in x:
                visit(item)
        else:
            digest.update(repr(x).encode())
        digest.update(b"\0")

    visit(value)
    return digest.hexdigest()


def snapshot(backend):
    views = model_slices(backend, state_filter)
    return {
        "model": {k: v.tensor.detach().cpu().clone() for k, v in views.items()},
        "optimizer": backend.optimizer.state_dict(),
        "rng": backend._rng_state(),
    }


def update(backend, inputs):
    # Consume all four RNG sources whose state must be restored.
    jitter = random.random() + float(np.random.random()) + float(torch.rand(()))
    loss = backend.module(inputs).float().square().mean() * (0.9 + jitter / 10)
    backend.backward(loss)
    backend.step()
    torch.cuda.synchronize()
    return loss.detach().cpu()


def maximum(value):
    t = torch.tensor(value, dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def run(output, strategy, offload):
    rank = dist.get_rank()
    case = f"{strategy}-offload{int(offload)}"
    root = output / case
    backend = create_backend(offload, strategy)
    torch.manual_seed(100 + rank)
    random.seed(200 + rank)
    np.random.seed(300 + rank)
    inputs = torch.randn(4, 17, device="cuda", dtype=torch.bfloat16)
    options = {
        "max_shared_memory_bytes": 16 << 20,
        "pipeline_max_inflight_bytes": 16 << 20,
        "staging_tile_bytes": 1 << 20,
        "load_worker_count": 2,
    }
    manager = CheckpointManager(
        str(root), "probe", nfc_options=options, max_checkpoints=2
    )
    controller = TrainerController(
        SimpleNamespace(
            backend=backend,
            strategy=SimpleNamespace(
                name="dspark", checkpoint_state_filter=state_filter
            ),
        ),
        run_id="probe",
        output_dir=str(root),
        checkpoint_manager=manager,
        checkpoint_extra={
            "specforge_step_options": (("objective_checkpointing", True),),
            "model_provenance": {"target_layers": (1, 3)},
        },
    )
    start_step = 0
    if strategy == "NO_SHARD":
        # DDP rebuilds reduction buckets after its first backward. A cold
        # first step and a replay on a warmed reducer can round differently.
        # Stabilize that derived runtime state before the bitwise comparison.
        assert not backend.module._has_rebuilt_buckets
        update(backend, inputs)
        update(backend, inputs)
        assert backend.module._has_rebuilt_buckets
        start_step = 2
    results = []
    try:
        # Include lazy Adam state at step 0; then reuse the same bounded pool.
        for iteration, step in enumerate(range(start_step, start_step + 4)):
            if iteration:
                update(backend, inputs)
            expected = fingerprint(snapshot(backend))
            controller._epoch_batch = step
            controller._epoch_samples = step * 4
            started = time.perf_counter()
            checkpoint = controller.save_checkpoint(step)
            elapsed = maximum(time.perf_counter() - started)
            manager.update_best(step, {"eval/simulated_acc_len": float(step)})
            # Exercise a real optimizer update before the disk/publication fence.
            # The captured checkpoint must remain immutable after source release.
            assert manager._nfc.pending is not None
            assert not Path(manager.checkpoint_dir(step)).exists()
            reference_loss = update(backend, inputs)
            reference = fingerprint(snapshot(backend))
            manager.finalize_pending()
            durable = maximum(time.perf_counter() - started)
            path = CheckpointManager.resolve_resume_dir(checkpoint.checkpoint_uri)
            assert manager.best_step == step
            metadata = CheckpointManager.read_resume_state(path)
            assert metadata["epoch_samples"] == step * 4
            assert metadata["global_step"] == step
            manager.restore_native(backend, state_filter, path)
            assert (
                fingerprint(snapshot(backend)) == expected
            ), f"{case} step{step}: saved state changed"
            resumed_loss = update(backend, inputs)
            assert torch.equal(reference_loss, resumed_loss), (
                reference_loss,
                resumed_loss,
            )
            assert (
                fingerprint(snapshot(backend)) == reference
            ), f"{case} step{step}: next step differs"
            results.append(
                {
                    "step": step,
                    "source_release_seconds": elapsed,
                    "durable_after_next_step_seconds": durable,
                    "next_step_bitwise": True,
                    "marks": dict(manager._nfc.runtime.last_save_marks),
                }
            )
        # Restore into a fresh wrapped model and lazy Adam optimizer, at a new
        # master placement. Then replay the exact next step again.
        manager.restore_native(backend, state_filter, path)
        expected_loss = update(backend, inputs)
        fresh = create_backend(not offload, strategy)
        manager.restore_native(fresh, state_filter, path)
        resumed_loss = update(fresh, inputs)
        assert torch.equal(expected_loss, resumed_loss)
        # Kernel placement can change roundoff: compare the same placement
        # bitwise above; a placement switch must retain saved masters/moments.
        manager.restore_native(backend, state_filter, path)
        manager.restore_native(fresh, state_filter, path)
        for a, b in zip(backend.optimizer.fp32_params, fresh.optimizer.fp32_params):
            assert torch.equal(a.cpu(), b.cpu())
        if rank == 0:
            exported = load_draft_weights(path)["draft_state_dict"]
            torch.save(exported, root / "export-reference.pt")
            assert exported["proj.weight"].shape == (29, 17)
            assert exported["fp32.scale"].dtype == torch.float32
            assert exported["empty"].numel() == 0
            assert set(exported) == set(model_slices(backend, state_filter))
        dist.barrier()
        exported = torch.load(
            root / "export-reference.pt", map_location="cpu", weights_only=True
        )
        for name, view in model_slices(backend, state_filter).items():
            expected_slice = exported[name].view(-1)[
                view.offset : view.offset + view.tensor.numel()
            ]
            assert torch.equal(
                expected_slice, view.tensor.detach().cpu().view(-1)
            ), name
        record = {
            "rank": rank,
            "case": case,
            "passed": True,
            "saves": results,
            "fresh_restore": True,
            "placement_switch": True,
            "ddp_warmup_steps": start_step,
        }
        (output / f"{case}-rank{rank}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        print(json.dumps(record), flush=True)
    finally:
        manager.close()
    del controller, backend
    gc.collect()
    torch.cuda.empty_cache()


def trainer_lifecycle(output):
    """Exercise the public Trainer's prewrap contract + postwrap native load."""
    from specforge.algorithms.common.providers import StepRuntimeConfig
    from specforge.runtime.control_plane import DataFlowController
    from specforge.runtime.data_plane.feature_store import LocalFeatureStore
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader
    from specforge.training.strategies.base import DraftTrainStrategy, StepOutput
    from specforge.training.trainer import Trainer

    rank = dist.get_rank()
    root = output / "trainer-lifecycle"
    features = root / "features"
    if rank == 0:
        features.mkdir(parents=True)
        for index in range(8):
            torch.save(
                {
                    "input_ids": torch.arange(17) + index,
                    "loss_mask": torch.ones(17),
                    "hidden_state": torch.zeros(1, 17, 4),
                    "aux_hidden_state": torch.zeros(1, 17, 12),
                },
                features / f"{index:04d}.ckpt",
            )
    dist.barrier()

    class Strategy(DraftTrainStrategy):
        name = "eagle3"
        required_features = frozenset({"x"})

        def __init__(self, wrapped):
            self.model = wrapped

        def trainable_module(self):
            return self.model

        def forward_loss(self, batch, ctx=None):
            jitter = random.random() + float(np.random.random()) + float(torch.rand(()))
            loss = self.model(batch.tensors["x"]).float().square().mean() * (
                0.9 + jitter / 10
            )
            return StepOutput(loss=loss, metrics={})

        def checkpoint_state_filter(self, state):
            return state_filter(state)

    def build(name, max_steps, resume=None):
        torch.manual_seed(42)
        model = Model().cuda()
        trainer = Trainer(
            algorithm_name="eagle3",
            make_step_strategy=lambda wrapped, target_head: Strategy(wrapped),
            controller=DataFlowController(name),
            store=LocalFeatureStore("features"),
            ref_source={
                "refs": OfflineManifestReader(str(features), run_id="data").read()
            },
            model=model,
            target_head=None,
            optimizer_factory=lambda draft: BF16Optimizer(
                draft, lr=1e-3, total_steps=20, warmup_ratio=0.1, offload_master=True
            ),
            run_id=name,
            output_dir=str(root / name),
            batch_size=2,
            accumulation_steps=1,
            num_epochs=1,
            max_steps=max_steps,
            total_steps=20,
            save_interval=0,
            logger=None,
            log_interval=50,
            collate_fn=lambda items: {"x": torch.stack([item["x"] for item in items])},
            per_sample_transform=lambda raw: {
                "x": raw["input_ids"].to("cuda", torch.bfloat16) / 17
            },
            durable_ack=False,
            resume_from=resume,
            strategy_kwargs=StepRuntimeConfig(
                options={},
                resume_contract={"tuple_contract": (("native", True),)},
                allowed_missing_checkpoint_keys=frozenset(),
            ),
            nfc_checkpoint={
                "max_shared_memory_bytes": 16 << 20,
                "pipeline_max_inflight_bytes": 16 << 20,
                "staging_tile_bytes": 1 << 20,
                "load_worker_count": 2,
            },
        )
        if resume is None:
            torch.manual_seed(100 + rank)
            random.seed(200 + rank)
            np.random.seed(300 + rank)
        return trainer

    full = build("full", 3)
    assert full.fit() == 3
    reference = fingerprint(snapshot(full._controller.core.backend))
    split = build("split", 1)
    assert split.fit() == 1
    saved = split._controller._checkpoint_manager().latest_dir()
    assert not (Path(saved) / "training_state.pt").exists()
    resumed = build("resumed", 3, saved)
    assert resumed.global_step == 1
    assert resumed.fit() == 3
    assert fingerprint(snapshot(resumed._controller.core.backend)) == reference
    record = {
        "rank": rank,
        "public_trainer_resume_bitwise": True,
        "final_durable": True,
    }
    (output / f"trainer-lifecycle-rank{rank}.json").write_text(
        json.dumps(record) + "\n"
    )
    print(json.dumps(record), flush=True)


def async_publication(output):
    """Hold rank 0's disk worker while every rank advances the optimizer."""
    rank = dist.get_rank()
    backend = create_backend(True, "FULL_SHARD")
    inputs = torch.ones(4, 17, device="cuda", dtype=torch.bfloat16)
    update(backend, inputs)
    expected = fingerprint(snapshot(backend))
    manager = CheckpointManager(
        str(output / "async-publication"),
        "async",
        nfc_options={
            "max_shared_memory_bytes": 16 << 20,
            "pipeline_max_inflight_bytes": 16 << 20,
            "staging_tile_bytes": 1 << 20,
            "load_worker_count": 2,
        },
    )
    native = manager._native_checkpoint()
    runtime = native._runtime()
    executor = runtime._writer
    gate = threading.Event()
    fail = False

    def submit(fn):
        def write():
            if rank == 0:
                assert gate.wait(120), "test did not release the disk worker"
            fn()
            if fail and rank == 0:
                raise OSError("injected rank-0 disk failure")

        return executor.submit(write)

    runtime._writer = SimpleNamespace(submit=submit)
    metadata = {
        "global_step": 1,
        "world_size": dist.get_world_size(),
        "run_id": "async",
    }
    try:
        path = manager.save_native(backend, state_filter, metadata, 1)
        manager.update_best(1, {manager.best_metric: 1.0})
        assert not manager.finalize_pending(wait=False)
        assert manager.latest_dir() is None
        assert manager.best_step is None
        update(backend, inputs)
        gate.set()
        manager.finalize_pending()
        assert manager.best_step == 1
        assert manager.latest_dir() == path
        manager.restore_native(backend, state_filter, path)
        assert fingerprint(snapshot(backend)) == expected
        previous_marker = (Path(path) / "training_state.json").read_bytes()
        fail = True
        update(backend, inputs)
        manager.save_native(backend, state_filter, metadata, 1)
        manager.update_best(1, {manager.best_metric: 2.0})
        try:
            manager.finalize_pending()
        except Exception as exc:
            assert "injected rank-0 disk failure" in str(exc), str(exc)
        else:
            raise AssertionError("all ranks must observe the disk failure")
        assert manager.best_step == 1
        assert manager.latest_dir() == path
        assert (Path(path) / "training_state.json").read_bytes() == previous_marker
        record = {
            "rank": rank,
            "partial_ready_did_not_publish": True,
            "optimizer_advanced_while_rank0_write_blocked": True,
            "snapshot_restored_bitwise": True,
            "rank0_failure_seen": True,
            "previous_same_step_checkpoint_preserved": True,
        }
        (output / f"async-publication-rank{rank}.json").write_text(
            json.dumps(record) + "\n"
        )
        print(json.dumps(record), flush=True)
    finally:
        gate.set()
        runtime._writer = executor
        try:
            manager.close()
        except Exception as exc:
            assert fail and "injected rank-0 disk failure" in str(exc), str(exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        for strategy, offload in [
            ("FULL_SHARD", True),
            ("SHARD_GRAD_OP", False),
            ("NO_SHARD", True),
        ]:
            run(args.output, strategy, offload)
        trainer_lifecycle(args.output)
        async_publication(args.output)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
