"""Capture semantic matrices before FSDP1 changes original parameter views."""

import math
from dataclasses import dataclass, field

from torch import nn


@dataclass
class LogicalParameter:
    name: str
    shape: tuple[int, ...]
    role: str
    matrix_shape: tuple[int, int] | None
    parameter: nn.Parameter = field(repr=False)
    offset: int = 0
    sharded: bool = False
    process_group: object = field(default=None, repr=False)

    def identity(self):
        return {
            "name": self.name,
            "shape": list(self.shape),
            "role": self.role,
            "matrix_shape": list(self.matrix_shape) if self.matrix_shape else None,
            "offset": self.offset,
            "numel": self.parameter.numel(),
            "sharded": self.sharded,
        }


def capture_layout(model, config):
    """Only module-owned Linear weights are generic matrices; unknowns use AdamW.

    Native layouts come from concrete module instances and their config, never
    from local shard ndim or suffix guesses. Each matrix is a contiguous region.
    """
    overrides = {}
    native = (
        getattr(getattr(model, "config", None), "model_type", None)
        == "deepseek_v41_draft"
    )
    if config.profile == "v41_flash_muon" and not native:
        raise ValueError("v41_flash_muon requires DeepSeekV41DraftModel")
    if native:
        from specforge.modeling.draft.deepseek_v41.model import Attention, MarkovHead
        from specforge.modeling.draft.deepseek_v41.ops import Gate, HCMappings, MoE

        for module in model.modules():
            if isinstance(module, Attention):
                c = module.c
                if config.profile == "v41_flash_muon":
                    overrides[id(module.wq_b.weight)] = (
                        "query_head",
                        (c.head_dim, c.q_lora_rank),
                    )
                    # Native wkv produces one shared KV head, not concatenated K/V.
                    overrides[id(module.wkv.weight)] = (
                        "shared_kv_head",
                        (c.head_dim, c.hidden_size),
                    )
                overrides[id(module.wo_a.weight)] = (
                    "output_group",
                    (c.o_lora_rank, c.num_attention_heads * c.head_dim // c.o_groups),
                )
            elif isinstance(module, MoE):
                e, twice_i, d = module.gate_up.shape
                if twice_i % 2 or tuple(module.down.shape) != (e, d, twice_i // 2):
                    raise ValueError("unexpected native MoE gate/up/down layout")
                overrides[id(module.gate_up)] = (
                    "expert_gate_up_split",
                    (twice_i // 2, d),
                )
                overrides[id(module.down)] = ("expert_down", (d, twice_i // 2))
            elif isinstance(module, MarkovHead):
                # DSpark-specific tables: explicit AdamW fallback, no claim of
                # reproducing the main model's Sinkhorn embedding optimizer.
                overrides.update({id(p): ("markov", None) for p in module.parameters()})
            elif isinstance(module, HCMappings):
                overrides[id(module.fn)] = ("mhc_linear", tuple(module.fn.shape))
                overrides[id(module.base)] = ("bias_or_scale", None)
                overrides[id(module.scale)] = ("bias_or_scale", None)
            elif isinstance(module, Gate):
                overrides[id(module.weight)] = (
                    "router_linear",
                    tuple(module.weight.shape),
                )
    owners = {}
    for module in model.modules():
        for key, param in module.named_parameters(recurse=False):
            role, matrix = "unknown_adamw", None
            if isinstance(module, nn.Embedding):
                role = "embedding"
            elif isinstance(module, nn.Linear) and key == "weight":
                role, matrix = "linear", tuple(param.shape)
            elif isinstance(module, nn.LayerNorm) or (
                native and type(module).__name__ == "RMSNorm"
            ):
                role = "normalization" if key == "weight" else "bias_or_scale"
            elif param.ndim < 2:
                role = "bias_or_scale"
            owners.setdefault(id(param), (role, matrix))
    result = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        role, matrix = overrides.get(id(param), owners[id(param)])
        if matrix is not None:
            if (
                len(matrix) != 2
                or min(matrix) <= 0
                or param.numel() % math.prod(matrix)
            ):
                raise ValueError(f"invalid logical matrix layout: {name}")
            if math.prod(matrix) > config.max_matrix_elements:
                raise ValueError(
                    f"Muon matrix exceeds max_matrix_elements: {name} {matrix}"
                )
        result.append(LogicalParameter(name, tuple(param.shape), role, matrix, param))
    if not result or not any(p.matrix_shape for p in result):
        raise ValueError("Muon requires at least one recognized trainable matrix")
    return result


def bind_fsdp_layout(plan, wrapped_model):
    """Bind original parameter coordinates to FSDP's documented-by-source metadata.

    Validate every nonempty interval once across ranks. Padding is never a
    logical parameter region. Only FULL_SHARD on a single WORLD group is supported.
    """
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    by_id = {id(item.parameter): item for item in plan}
    for fsdp in FSDP.fsdp_modules(wrapped_model):
        if not fsdp._use_orig_params:
            raise ValueError("Muon requires FSDP use_orig_params=True")
        if fsdp.sharding_strategy != ShardingStrategy.FULL_SHARD:
            raise ValueError("Muon FSDP requires FULL_SHARD; use DDP for replication")
        handle = fsdp._handle
        if handle is None:
            continue
        if handle.world_size != dist.get_world_size() or handle.rank != dist.get_rank():
            raise ValueError("Muon FSDP requires the WORLD process group")
        flat = handle.flat_param
        for param, shape, info in zip(
            flat._params, flat._shapes, flat._shard_param_infos, strict=True
        ):
            item = by_id.get(id(param))
            if item is None:
                continue
            if tuple(shape) != item.shape:
                raise ValueError(f"original FSDP shape changed: {item.name}")
            expected = info.numel_in_shard if info.in_shard else 0
            if param.numel() != expected:
                raise ValueError(
                    f"Muon must be built at a sharded boundary: {item.name}"
                )
            item.offset = info.intra_param_start_idx if info.in_shard else 0
            item.sharded, item.process_group = True, fsdp.process_group
    local = [item.identity() for item in plan]
    if dist.is_initialized():
        layouts = [None] * dist.get_world_size()
        dist.all_gather_object(layouts, local)
    else:
        layouts = [local]
    for index, item in enumerate(plan):
        entries = [layout[index] for layout in layouts if len(layout) == len(local)]
        if len(entries) != len(layouts):
            raise ValueError("Muon parameter count differs across ranks")
        semantic = {
            k: v for k, v in local[index].items() if k not in {"offset", "numel"}
        }
        if any(
            {k: v for k, v in e.items() if k not in {"offset", "numel"}} != semantic
            for e in entries
        ):
            raise ValueError("Muon logical layout differs across ranks")
        if item.sharded:
            cursor = 0
            for start, count in sorted(
                (e["offset"], e["numel"]) for e in entries if e["numel"]
            ):
                if start != cursor:
                    raise ValueError(f"Muon shard gap/overlap: {item.name}")
                cursor += count
            if cursor != math.prod(item.shape):
                raise ValueError(f"Muon incomplete shard coverage: {item.name}")
        elif any(e["numel"] != math.prod(item.shape) for e in entries):
            raise ValueError(f"missing Muon FSDP metadata: {item.name}")
    return plan
