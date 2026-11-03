"""torchrun --nproc_per_node=8: unequal supervision, native FP32 head, FULL_SHARD."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from specforge.optimizer import BF16Optimizer
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from tests.test_modeling.test_dspark_anchor_padding import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    rank = dist.get_rank()
    assert dist.get_world_size() == 8
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        for checkpointing in (False, True):
            torch.manual_seed(42)
            model = build_model().cuda().bfloat16()
            model.objective_checkpointing = checkpointing
            head = model.draft_model.confidence_head
            head.float()
            calls = [0]

            def count(*_):
                calls[0] += 1

            hook = head.register_forward_pre_hook(count)
            backend = FSDPTrainingBackend(
                ParallelConfig(world_size=8, sharding_strategy="FULL_SHARD",
                               fsdp_process_group=dist.group.WORLD),
                optimizer_factory=lambda draft: BF16Optimizer(
                    draft, lr=1e-4, total_steps=2, warmup_ratio=0, offload_master=True),
            )
            backend.prepare_model(model, optimizer_target=model.draft_model)
            for micro in range(32):
                valid = (1, 7, 8, 9, 19, 24, 25, 48)[(rank + micro) % 8]
                torch.manual_seed(1000 + rank * 32 + micro)
                before = calls[0]
                boundary = (micro + 1) % 16 == 0
                with backend.accumulation_context(is_boundary=boundary):
                    loss, _, _ = backend.module(
                        input_ids=torch.randint(0, 30, (1, valid + 1), device="cuda"),
                        loss_mask=torch.ones(1, valid + 1, device="cuda"),
                        hidden_states=torch.randn(1, valid + 1, 16, device="cuda", dtype=torch.bfloat16),
                        target_last_hidden_states=torch.randn(1, valid + 1, 16, device="cuda", dtype=torch.bfloat16),
                        max_valid_anchors=valid,
                    )
                    assert calls[0] - before == 4
                    assert torch.isfinite(loss)
                    backend.backward(loss / 16, is_boundary=boundary)
                if boundary:
                    norm = backend.step()
                    assert norm is not None and torch.isfinite(norm) and norm > 0
                counts = [None] * 8
                dist.all_gather_object(counts, calls[0] - before)
                assert len(set(counts)) == 1, counts
            hook.remove()
            records.append({"checkpointing": checkpointing, "optimizer_steps": 2,
                            "micro_steps": 32, "head_calls": calls[0],
                            "all_ranks_same_forward_and_replay_calls": True})
        receipt = {"passed": True, "rank": rank, "world_size": 8,
                   "torch": torch.__version__, "cases": records,
                   "scope": "real FULL_SHARD, native FP32 ConfidenceHead, DSpark objective and CPU-offloaded optimizer"}
        (args.output / f"rank{rank}.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
