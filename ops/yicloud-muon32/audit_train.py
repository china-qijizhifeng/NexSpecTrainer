"""Run the full CLI while auditing actual exhaustive-anchor execution."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import psutil
import math
from collections import Counter
from fixed_reduction import install

install()
from specforge.algorithms.deepseek_v41.strategy import NativeStrategy
from specforge.training.strategies.base import DSparkTrainStrategy
from specforge.training.backend import FSDPTrainingBackend
from specforge.training.controller import TrainerCore
from specforge.cli import main as cli


def main():
    root = Path(os.environ["TASK_ROOT"])
    out = root / "training"
    out.mkdir(exist_ok=True)
    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 32
    expected = {
        v["input_sha256"]: v
        for v in json.loads((Path(os.environ["INPUT_ROOT"]) / "data-audit.json").read_text())["windows"].values()
    }
    state = dict(
        rank=rank,
        optimizer_steps=0,
        training_micro=0,
        eval_micro=0,
        eligible_anchors=0,
        target_exposures=0,
        start_time=time.time(),
    )
    watched, consumed = {}, []

    def save(name, value):
        path = out / name
        tmp = path.with_suffix(".partial")
        tmp.write_text(json.dumps(value, indent=2) + "\n")
        tmp.replace(path)

    def digest(tensor):
        h = hashlib.sha256()
        flat = tensor.detach().reshape(-1)
        for i in range(0, flat.numel(), 4 << 20):
            h.update(
                flat[i : i + (4 << 20)].cpu().contiguous().view(torch.uint8).numpy()
            )
        return h.hexdigest()

    prepare = FSDPTrainingBackend.prepare_model

    def audited_prepare(self, model, **kw):
        wrapped = prepare(self, model, **kw)
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        assert all(
            m.mixed_precision.reduce_dtype == torch.float32
            for m in wrapped.modules()
            if isinstance(m, FSDP)
        )
        assert (
            model.anchor_sampling == "all"
            and model.anchor_forward_chunk_blocks
            == json.loads((Path(__file__).parent / "performance-selection.json").read_text())[
                "selected_chunk"
            ]
        )
        assert self._sync_each_microstep and not model.fsdp_prefetch
        assert (
            model.draft_model.config.gradient_checkpointing
            == json.loads((Path(__file__).parent / "performance-selection.json").read_text())[
                "selected_gradient_checkpointing"
            ]
        )
        from specforge.muon import MuonAdamWOptimizer
        assert isinstance(self.optimizer, MuonAdamWOptimizer)
        assert self.optimizer.muon_config.profile == "v41_flash_muon"
        assert {g["algorithm"] for g in self.optimizer.optimizer.param_groups} == {"muon", "adamw"}
        save(f"optimizer-identity-rank{rank}.json", self.optimizer.checkpoint_identity())
        roles = Counter()
        for item in self.optimizer.plan:
            roles[item.role] += math.prod(item.shape)
        save(f"optimizer-groups-rank{rank}.json", dict(roles))
        assert self.optimizer.offload_master
        frozen = [p for m in self.ignored_frozen_modules for p in m.parameters()]
        assert len(frozen) == 2 and all(
            not p.requires_grad and p.grad is None for p in frozen
        )
        assert {id(p) for p in frozen}.isdisjoint(
            id(p) for p in self.optimizer.model_params
        )
        watched.update(frozen=frozen, hashes=[digest(p) for p in frozen])
        save(
            f"frozen-baseline-rank{rank}.json",
            dict(hashes=watched["hashes"], excluded_from_optimizer=True),
        )
        return wrapped

    FSDPTrainingBackend.prepare_model = audited_prepare

    def audit(self, batch, invoke, train):
        sha = digest(batch.tensors["input_ids"])
        row = expected[sha]
        mask = batch.tensors["loss_mask"].detach().cpu()
        assert (
            hashlib.sha256((mask > 0.5).numpy().astype("u1").tobytes()).hexdigest()
            == row["mask_sha256"]
        )
        assert row["split"] == ("train" if train else "validation")
        state["training_micro" if train else "eval_micro"] += 1
        save(
            f"micro-progress-rank{rank}.json",
            {**state, "phase": "started", "time": time.time()},
        )
        result = invoke()
        assert torch.isfinite(result.loss).item()
        counts = {k: int(v.item()) for k, v in result.sum_metrics.items()}
        assert (
            counts["anchors/selected"]
            == counts["anchors/eligible"]
            == row["eligible_anchors"]
        )
        assert counts["anchors/duplicates"] == 0
        assert counts["supervision/target_exposures"] == row["target_exposures"]
        for k, n in enumerate(row["position_exposures"], 1):
            assert counts[f"supervision/position_{k}"] == n
        if train:
            consumed.append(sha)
            state["eligible_anchors"] += row["eligible_anchors"]
            state["target_exposures"] += row["target_exposures"]
        watched["loss"] = float(result.loss)
        watched["counts"] = counts
        return result

    stream = NativeStrategy.stream_backward

    def audited_stream(self, batch, backward, ctx=None):
        return audit(self, batch, lambda: stream(self, batch, backward, ctx), True)

    NativeStrategy.stream_backward = audited_stream
    forward = DSparkTrainStrategy.forward_loss

    def audited_forward(self, batch, ctx=None):
        assert not self.trainable_module().training
        return audit(self, batch, lambda: forward(self, batch, ctx), False)

    DSparkTrainStrategy.forward_loss = audited_forward

    step = FSDPTrainingBackend.step

    def audited_step(self, **kw):
        lr = self.optimizer.optimizer.param_groups[0]["lr"]
        torch.cuda.synchronize()
        optimizer_started = time.perf_counter()
        norm = step(self, **kw)
        torch.cuda.synchronize()
        optimizer_seconds = time.perf_counter() - optimizer_started
        assert norm is not None and torch.isfinite(norm).item()
        assert all(p.grad is None for p in watched["frozen"])
        state["optimizer_steps"] += 1
        record = {
            **state,
            "time": time.time(),
            "grad_norm": float(norm),
            "optimizer_seconds": optimizer_seconds,
            "host_rss_bytes": psutil.Process().memory_info().rss,
            "optimizer_algorithms": sorted({g["algorithm"] for g in self.optimizer.optimizer.param_groups}),
            "lr_used": lr,
            "lr_next": self.optimizer.optimizer.param_groups[0]["lr"],
        }
        save(f"live-progress-rank{rank}.json", record)
        with (out / f"steps-rank{rank}.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        return norm

    FSDPTrainingBackend.step = audited_step

    train_step = TrainerCore.train_step

    def measured_micro(self, batch, ctx=None):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        result = train_step(self, batch, ctx)
        torch.cuda.synchronize()
        assert torch.cuda.max_memory_allocated() < 65 * 2**30
        record = {
            **state,
            "phase": "complete",
            "time": time.time(),
            "seconds": time.time() - start,
            "local_loss": watched["loss"],
            "coverage": watched["counts"],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "allocated_bytes": torch.cuda.memory_allocated(),
        }
        save(f"micro-progress-rank{rank}.json", record)
        with (out / f"microbatches-rank{rank}.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        if state["training_micro"] == 5:
            assert state["optimizer_steps"] == 1
            assert [digest(p) for p in watched["frozen"]] == watched["hashes"]
            save(f"startup-acceptance-rank{rank}.json", {
                "passed": True, "profile": "v41_flash_muon", "all_anchors": True,
                "optimizer_updates": 1, "training_micro": 5,
                "post_update_backward_finite": True, "frozen_unchanged": True,
                "scope": "startup only; not convergence or checkpoint restore acceptance",
            })
        return result

    TrainerCore.train_step = measured_micro

    code = cli(sys.argv[1:])
    assert (
        code == 0
        and state["optimizer_steps"] == 118
        and state["training_micro"] == 472
    )
    assert [digest(p) for p in watched["frozen"]] == watched["hashes"]
    save(
        f"complete-rank{rank}.json",
        {
            **state,
            "passed": True,
            "consumed_input_sha256": consumed,
            "frozen_unchanged": True,
        },
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
