"""Opt-in diagnostic instrumentation for pinned SGLang 38d9e4a3f.

Control via PREFILL_ROOT/control.json. Synchronization is ONLY used when
profile=true. Production throughput must be measured with profile=false.
"""

import functools
import json
import os
import time
from pathlib import Path


def install(scheduler):
    import torch
    from sglang.srt.layers.engram import Engram, EngramEmbedding
    from sglang.srt.runtime_context import get_parallel

    runner = scheduler.tp_worker.model_runner
    root = Path(os.environ["PREFILL_ROOT"])
    rank = get_parallel().tp_rank
    state = {"active": False, "events": [], "sequence": 0, "runtime_mode": "baseline"}

    def timed_method(cls, name, label):
        original = getattr(cls, name)

        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            if not state["active"]:
                return original(self, *args, **kwargs)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            begin = time.perf_counter()
            result = original(self, *args, **kwargs)
            host_ms = (time.perf_counter() - begin) * 1000
            end.record()
            state["events"].append((label, start, end, host_ms))
            return result

        setattr(cls, name, wrapped)

    timed_method(EngramEmbedding, "_owned_rows", "engram_host_gather")
    timed_method(EngramEmbedding, "_reduce_owned_rows", "engram_all_reduce")
    timed_method(Engram, "forward", "engram_total")
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
    from sglang.srt.models.deepseek_v4 import MQALayer

    timed_method(DeepseekV2MoE, "forward", "moe_total")
    timed_method(MQALayer, "forward", "attention_total")
    from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
        DenseBlocksBackend,
    )
    from sglang.srt.layers.attention.dsv4.v41_indexer.full_topk import FullTopKIndexer

    timed_method(FullTopKIndexer, "topk_prefill", "indexer_full")
    timed_method(DenseBlocksBackend, "publish_prefill", "indexer_publish")
    timed_method(DenseBlocksBackend, "consume_prefill", "indexer_consume")
    original = runner._forward_raw

    @functools.wraps(original)
    def forward(batch, *args, **kwargs):
        cfg = json.loads((root / "control.json").read_text())
        mode = cfg.get("runtime_mode", "baseline")
        if mode != state["runtime_mode"]:
            from runtime_control import configure

            configure(runner, mode)
            state["runtime_mode"] = mode
        if not cfg.get("profile") or not batch.forward_mode.is_extend():
            return original(batch, *args, **kwargs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        memory_before = torch.cuda.memory_allocated()
        state["events"] = []
        state["active"] = cfg.get("detail", False)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        begin = time.perf_counter()
        start.record()
        try:
            result = original(batch, *args, **kwargs)
        finally:
            end.record()
            end.synchronize()
            state["active"] = False
        wall_ms = (time.perf_counter() - begin) * 1000
        groups = {}
        for name, a, b, host in state["events"]:
            item = groups.setdefault(name, {"cuda_ms": 0.0, "host_ms": 0.0, "calls": 0})
            item["cuda_ms"] += a.elapsed_time(b)
            item["host_ms"] += host
            item["calls"] += 1
        record = {
            "rank": rank,
            "sequence": state["sequence"],
            "tag": cfg.get("tag"),
            "tokens": int(batch.input_ids.numel()),
            "forward_cuda_ms": start.elapsed_time(end),
            "forward_wall_ms": wall_ms,
            "groups": groups,
            "gpu_allocated_before": memory_before,
            "gpu_peak_allocated": torch.cuda.max_memory_allocated(),
        }
        state["sequence"] += 1
        with (root / f"forward-rank{rank}.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        return result

    runner._forward_raw = forward
    # Measure copy execution, not just asynchronous submission.
    if rank == 0:
        from specforge_sglang_capture.capture import CaptureDeviceOutput

        observer = scheduler._specforge_capture_runtime.observer
        original_observe = observer.after_forward

        def observe(batch, forward_batch, logits_output, **kwargs):
            cfg = json.loads((root / "control.json").read_text())
            if cfg.get("profile"):
                with (root / "queue.jsonl").open("a") as f:
                    for req in batch.reqs:
                        f.write(
                            json.dumps(
                                {
                                    "tag": cfg.get("tag"),
                                    "rid": req.rid,
                                    "queue_s": req.time_stats.get_queueing_time(),
                                    "scheduler_receive_to_forward_s": req.time_stats.forward_entry_time
                                    - req.time_stats.scheduler_recv_time,
                                }
                            )
                            + "\n"
                        )
            return original_observe(batch, forward_batch, logits_output, **kwargs)

        observer.after_forward = observe
        old_copy = CaptureDeviceOutput.copy_to_host

        @functools.wraps(old_copy)
        def copy(self, copy_tensor):
            cfg = json.loads((root / "control.json").read_text())
            if not cfg.get("profile"):
                return old_copy(self, copy_tensor)
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            a.record()
            begin = time.perf_counter()
            result = old_copy(self, copy_tensor)
            b.record()
            b.synchronize()
            with (root / "copies.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "tag": cfg.get("tag"),
                            "cuda_ms": a.elapsed_time(b),
                            "wall_ms": (time.perf_counter() - begin) * 1000,
                        }
                    )
                    + "\n"
                )
            return result

        CaptureDeviceOutput.copy_to_host = copy
