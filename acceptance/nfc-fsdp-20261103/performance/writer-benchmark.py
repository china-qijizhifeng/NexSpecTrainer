"""Real dSpark writer sweep at 32GiB/rank. Original inputs are read-only."""

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from specforge.modeling.draft.deepseek_v41 import (
    DeepSeekV41DraftConfig,
    DeepSeekV41DraftModel,
)
from specforge.optimizer import BF16Optimizer
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.checkpoint import CheckpointManager
from specforge.training.controller import TrainerController
from specforge.training.nfc_state import NativeState

OLD = Path(
    "/gpfs/users/linlizhi/nexspec-v41-rdma-realtrace-20261103/training-r1/v41-native-real-features-r1-step2"
)
RANK = int(os.environ["RANK"])


def log(phase, **kw):
    print(json.dumps(dict(phase=phase, rank=RANK, **kw)), flush=True)


def maximum(x):
    t = torch.tensor(x, dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def fingerprint(state):
    h = hashlib.sha256()
    count = size = 0

    def visit(x):
        nonlocal count, size
        h.update(type(x).__qualname__.encode())
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().contiguous()
            h.update(str((str(t.dtype), tuple(t.shape))).encode())
            raw = t.reshape(-1).view(torch.uint8).numpy()
            for start in range(0, raw.size, 16 << 20):
                h.update(memoryview(raw[start : start + (16 << 20)]))
            count += 1
            size += t.numel() * t.element_size()
        elif isinstance(x, dict):
            for key, value in x.items():
                visit(key)
                visit(value)
        elif isinstance(x, (tuple, list)):
            for item in x:
                visit(item)
        else:
            h.update(repr(x).encode())
        h.update(b"\0")

    visit(state)
    return {"sha256": h.hexdigest(), "tensor_count": count, "tensor_bytes": size}


def state_filter(state):
    return {
        k.removeprefix("draft_model."): v
        for k, v in state.items()
        if k.startswith("draft_model.")
    }


def digest(backend):
    state = NativeState(backend, state_filter)
    # BF16Optimizer state contains the exact live CPU masters/moments, no clone.
    return fingerprint(
        {
            "model": {k: v.tensor for k, v in state.model.items()},
            "optimizer": backend.optimizer.state_dict(),
            "rng": backend._rng_state(),
        }
    )


def fsync_checkpoint(path):
    for p in path.rglob("*.pt"):
        if p.name == f"training_state_rank{RANK}.pt" or (
            RANK == 0 and p.name == "training_state.pt"
        ):
            with p.open("rb") as f:
                os.fsync(f.fileno())
    if RANK == 0:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    dist.barrier()


def make_controller(backend, root, label, options, metadata):
    manager = CheckpointManager(str(root / label), label, nfc_options=options)
    controller = TrainerController(
        SimpleNamespace(
            backend=backend,
            strategy=SimpleNamespace(
                name="deepseek_v41", checkpoint_state_filter=state_filter
            ),
        ),
        run_id=label,
        output_dir=str(root / label),
        checkpoint_manager=manager,
        checkpoint_extra=metadata,
    )
    controller._epoch_samples = 2
    controller._epoch_batch = 2
    return manager, controller


def perturb_live_state(backend):
    # Only private live mappings are mutated; original checkpoint files stay intact.
    import mmap
    import random

    import numpy as np

    assert torch.serialization.get_default_mmap_options() == mmap.MAP_PRIVATE
    tensors = [
        view.tensor for view in NativeState(backend, state_filter).model.values()
    ]
    tensors.extend(backend.optimizer.fp32_params)
    for state in backend.optimizer.optimizer.state.values():
        tensors.extend(
            value for value in state.values() if isinstance(value, torch.Tensor)
        )
    with torch.no_grad():
        for tensor in tensors:
            if tensor.numel():
                tensor.view(-1)[0] = 17
    backend.optimizer.scheduler.last_epoch += 13
    backend.optimizer.optimizer.param_groups[0]["lr"] = 0.123
    torch.manual_seed(99100 + RANK)
    torch.cuda.manual_seed(99200 + RANK)
    random.seed(99300 + RANK)
    np.random.seed(99400 + RANK)


def instrument_runtime(runtime, telemetry):
    original_save = runtime.save

    def measured_save(*args, **kwargs):
        started = time.perf_counter()
        ticket = original_save(*args, **kwargs)
        telemetry["runtime_save_return_seconds"] = time.perf_counter() - started
        original_wait = ticket.wait

        def measured_wait(*args, **kwargs):
            result = original_wait(*args, **kwargs)
            telemetry["runtime_recoverable_seconds"] = time.perf_counter() - started
            return result

        ticket.wait = measured_wait
        return ticket

    runtime.save = measured_save
    return original_save


def checkpoint_files(path):
    files = [p for p in path.rglob("*") if p.is_file()]
    return {
        "file_count": len(files),
        "file_bytes": sum(p.stat().st_size for p in files),
    }


def main(output):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(4)
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 8
    output.mkdir(parents=True, exist_ok=True)
    workspace = output / "payload"
    assert not workspace.exists(), "use a fresh output directory"
    dist.barrier()
    original_manifest = {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in OLD.glob("*.pt")
    }
    assert len(original_manifest) == 9
    free = os.statvfs(output)
    assert free.f_bavail * free.f_frsize > 300_000_000_000
    log("loading_inputs")
    shared = torch.load(
        OLD / "training_state.pt", map_location="cpu", weights_only=False, mmap=True
    )
    local = torch.load(
        OLD / f"training_state_rank{RANK}.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    assert (
        shared["strategy"] == "deepseek_v41"
        and shared["world_size"] == 8
        and shared["global_step"] == 2
    )
    cfg = DeepSeekV41DraftConfig.from_json_file(
        str(Path(__file__).with_name("deepseek-v41-flash-native.json"))
    )
    with torch.device("meta"):
        draft = DeepSeekV41DraftModel(cfg)
    draft.load_state_dict(shared["draft_state_dict"], strict=True, assign=True)
    numel = sum(p.numel() for p in draft.parameters())
    assert 14_000_000_000 < numel < 14_500_000_000, numel
    draft.cuda()
    model = torch.nn.Module()
    model.add_module("draft_model", draft)
    saved_opt = local["optimizer"]
    max_grad_norm = saved_opt["max_grad_norm"]
    scheduler_type = saved_opt.get("lr_scheduler_type", "cosine")
    backend = FSDPTrainingBackend(
        ParallelConfig(
            world_size=8,
            sharding_strategy="FULL_SHARD",
            fsdp_process_group=dist.group.WORLD,
        ),
        optimizer_factory=lambda m: BF16Optimizer(
            m,
            lr=1e-5,
            total_steps=10,
            max_grad_norm=max_grad_norm,
            lr_scheduler=scheduler_type,
            offload_master=True,
        ),
    )
    backend.prepare_model(model, optimizer_target=draft)
    for i, (saved, current) in enumerate(
        zip(saved_opt["fp32_params"], backend.optimizer.fp32_params, strict=True)
    ):
        assert saved.shape == current.shape, (RANK, i, saved.shape, current.shape)
    backend.optimizer.load_state_dict(saved_opt)
    backend._set_rng_state(local["rng"])
    metadata = {
        k: v
        for k, v in shared.items()
        if k
        not in {
            "draft_state_dict",
            "global_step",
            "epoch",
            "epoch_batch",
            "epoch_samples",
            "strategy",
            "run_id",
            "world_size",
        }
    }
    del shared, local, saved_opt
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    log(
        "inputs_ready",
        params=numel,
        gpu_bytes=torch.cuda.memory_allocated(),
        local_state=digest(backend),
    )
    expected = digest(backend)
    # Test actual quota after loading, immediately before the first save.
    dist.barrier()
    if RANK == 0:
        probe = output / "quota-reservation.bin"
        try:
            with probe.open("xb") as stream:
                os.posix_fallocate(stream.fileno(), 0, 220_000_000_000)
                os.fsync(stream.fileno())
                log("quota_reservation_passed", bytes=probe.stat().st_blocks * 512)
        finally:
            if probe.exists():
                probe.unlink()
    dist.barrier()

    def forbid_gather():
        raise AssertionError("native checkpoint called FULL_STATE_DICT")

    backend._module_state_dict = forbid_gather
    managers = {}
    records = []
    # Construct one runtime at a time: each rank-sized pool needs about 23.2GiB.
    # Releasing a case before constructing the next avoids accumulating pools.
    sequence = [
        (writers, phase)
        for writers in (4, 8, 16)
        for phase in ("cold", "warm1", "warm2")
    ]
    try:
        for index, (writers, phase) in enumerate(sequence):
            mode, gib = "native", 32
            if phase == "cold":
                opts = {
                    "max_shared_memory_bytes": gib << 30,
                    "pipeline_max_inflight_bytes": gib << 30,
                    "staging_tile_bytes": 128 << 20,
                    "writer_thread_count": writers,
                    "load_worker_count": 16,
                }
                managers[writers] = make_controller(
                    backend, workspace, f"native32g-w{writers}", opts, metadata
                )
            manager, controller = managers[writers]
            backend._module_state_dict = forbid_gather
            native = manager._native_checkpoint()
            telemetry = {}
            dist.barrier()
            torch.cuda.synchronize()
            log("save_begin", mode=mode, gib=gib, writers=writers, iteration=phase)
            start = time.perf_counter()
            if native is not None:
                runtime = native._runtime()
                telemetry["runtime_init_seconds"] = time.perf_counter() - start
                original_save = instrument_runtime(runtime, telemetry)
            ckpt = controller.save_checkpoint(2)
            returned = maximum(time.perf_counter() - start)
            path = Path(CheckpointManager.resolve_resume_dir(ckpt.checkpoint_uri))
            durable = maximum(time.perf_counter() - start)
            record = {
                "mode": mode,
                "pool_gib_per_rank": gib,
                "writers_per_rank": writers,
                "iteration": phase,
                "caller_pause_seconds": returned,
                "save_plus_fsync_seconds": durable,
                "parameters": numel,
                "telemetry": telemetry,
            }
            if native is not None:
                runtime.save = original_save
                record["marks"] = runtime.last_save_marks
                record["pipeline_stats"] = runtime._strategy._nfc_pipeline_writer.stats
                # Keep native writer telemetry without assuming its exact filename.
                record["native_json_files"] = {}
                for p in (path / "nfc").rglob("*.json"):
                    if (
                        "rank" in p.name or "pipeline" in p.name
                    ) and p.stat().st_size < (2 << 20):
                        record["native_json_files"][str(p.relative_to(path))] = (
                            json.loads(p.read_text())
                        )
            if RANK == 0:
                record.update(checkpoint_files(path))
            log("save_complete", **record)
            (
                output / f"{index:02d}-{mode}-{gib}g-w{writers}-{phase}-rank{RANK}.json"
            ).write_text(json.dumps(record, indent=2) + "\n")
            # Restore one warm checkpoint at each budget and compare every tensor.
            if phase == "warm2":
                perturb_live_state(backend)
                assert digest(backend) != expected, "restore test must reject a no-op"
                start = time.perf_counter()
                manager.restore_native(backend, state_filter, str(path))
                record["restore_seconds"] = maximum(time.perf_counter() - start)
                actual = digest(backend)
                assert actual == expected, (RANK, actual, expected)
                record["full_state_restored_bitwise"] = True
                (
                    output
                    / f"{index:02d}-{mode}-{gib}g-w{writers}-{phase}-rank{RANK}.json"
                ).write_text(json.dumps(record, indent=2) + "\n")
                log(
                    "restore_verified",
                    gib=gib,
                    writers=writers,
                    seconds=record["restore_seconds"],
                )
            records.append(record)
            dist.barrier()
            if RANK == 0:
                assert path.resolve().is_relative_to(workspace.resolve())
                shutil.rmtree(path)
            dist.barrier()
            if phase == "warm2":
                manager.close()
                gc.collect()
            assert {
                p.name: (p.stat().st_size, p.stat().st_mtime_ns)
                for p in OLD.glob("*.pt")
            } == original_manifest
        (output / f"passed-rank{RANK}.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "rank": RANK,
                    "records": len(records),
                    "expected": expected,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        for manager, _ in managers.values():
            manager.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
