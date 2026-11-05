"""Eight-rank optimizer differential on actual native head/expert parameter layouts.

No forward surrogate is used: this isolates optimizer semantics with identical
prescribed full gradients, including zeros for an expert, after real FSDP wrapping.
Native forward/backward is covered separately by the full-model CLI pilot.
"""

import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from specforge.config.schema import TrainingConfig
from specforge.modeling.draft.deepseek_v41.config import DeepSeekV41DraftConfig
from specforge.modeling.draft.deepseek_v41.model import DeepSeekV41DraftModel
from specforge.muon import MuonAdamWOptimizer
from specforge.muon_config import MuonConfig
from specforge.muon_layout import capture_layout
from specforge.training.assembly import _ConfiguredOptimizerFactory
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig


def run(dtype, offload):
    config = DeepSeekV41DraftConfig(
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
    torch.manual_seed(123)
    model = DeepSeekV41DraftModel(config).bfloat16().cuda()
    for module in model.modules():
        if getattr(module, "_specforge_keep_fp32", False):
            module.float()
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.normal_(std=0.02)
    reference = copy.deepcopy(model)
    options = MuonConfig(profile="v41_flash_muon", ns_dtype=dtype)
    training = TrainingConfig(
        optimizer="v41_flash_muon",
        muon=options,
        fsdp_sharding="FULL_SHARD",
        learning_rate=0.003,
        total_steps=20,
        warmup_ratio=0.1,
        max_grad_norm=0.1,
        optimizer_cpu_offload=offload,
    )
    backend = FSDPTrainingBackend(
        ParallelConfig(
            world_size=dist.get_world_size(),
            sharding_strategy="FULL_SHARD",
            fsdp_process_group=dist.group.WORLD,
        ),
        optimizer_factory=_ConfiguredOptimizerFactory(
            SimpleNamespace(training=training)
        ),
    )
    backend.prepare_model(model, optimizer_target=model)
    reference_plan = capture_layout(reference, options)
    optimizer = MuonAdamWOptimizer(
        reference,
        plan=reference_plan,
        config=options,
        lr=0.003,
        total_steps=20,
        warmup_ratio=0.1,
        max_grad_norm=0.1,
        offload_master=offload,
    )
    optimizer.configure_grad_norm_reduction(enabled=False)
    maximum = 0.0
    tolerance = 2e-6 if dtype == "float32" else 2e-4
    for step in range(4):
        generator = torch.Generator(device="cuda").manual_seed(200 + step)
        for item, reference_item in zip(
            backend.optimizer.plan, reference_plan, strict=True
        ):
            full_grad = torch.randn(
                item.shape,
                device="cuda",
                dtype=reference_item.parameter.dtype,
                generator=generator,
            )
            if len(item.shape) == 3:
                full_grad[step % item.shape[0]].zero_()
            reference_item.parameter.grad = full_grad
            item.parameter.grad = (
                full_grad.reshape(-1)[
                    item.offset : item.offset + item.parameter.numel()
                ]
                .view_as(item.parameter)
                .clone()
            )
        norm = backend.step()
        reference_norm = optimizer.step()
        assert norm > 0.1 and reference_norm > 0.1
        for item, master, full_master in zip(
            backend.optimizer.plan,
            backend.optimizer.fp32_params,
            optimizer.fp32_params,
            strict=True,
        ):
            expected = full_master.reshape(-1)[
                item.offset : item.offset + master.numel()
            ].view_as(master)
            torch.testing.assert_close(master, expected, atol=tolerance, rtol=tolerance)
            if master.numel():
                maximum = max(maximum, float((master - expected).abs().max()))
    return dict(
        ns_dtype=dtype,
        offload=offload,
        max_abs_error=maximum,
        tolerance=tolerance,
        roles=sorted({p.role for p in backend.optimizer.plan}),
        empty_shards=sum(p.parameter.numel() == 0 for p in backend.optimizer.plan),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    from specforge.distributed import _distributed_backend

    dist.init_process_group(_distributed_backend("cuda"))
    try:
        result = dict(
            rank=dist.get_rank(),
            cases=[run("float32", False), run("bfloat16", True)],
            passed=True,
        )
        args.output.mkdir(exist_ok=True, parents=True)
        (args.output / f"native-layout-rank{dist.get_rank()}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()
