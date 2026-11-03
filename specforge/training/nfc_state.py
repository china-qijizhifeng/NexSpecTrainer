"""FSDP1/use_orig_params -> native Core shards, without a model all-gather.

FSDP's flat-parameter metadata is the source of each original parameter's
flattened global coordinates. This adapter deliberately requires the same
world size and wrapping layout on restore. No full state_dict is requested.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.distributed as dist

from specforge.optimizer import BF16Optimizer

FORMAT = "specforge-nfc-fsdp-v1"
MANIFEST = "training_state.json"


def _clean_name(name):
    return ".".join(p for p in name.split(".") if p != "_fsdp_wrapped_module")


@dataclass
class TensorSlice:
    tensor: torch.Tensor
    shape: tuple
    offset: int
    replica: int
    buffer_owner: tuple | None = None

    def layout(self):
        return {
            "shape": list(self.shape),
            "offset": self.offset,
            "numel": self.tensor.numel(),
            "dtype": str(self.tensor.dtype),
            "replica": self.replica,
            "buffer": self.buffer_owner is not None,
        }

    def descriptor(self, key, tensor=None):
        from megatron.core.dist_checkpointing.mapping import ShardedTensor

        tensor = self.tensor if tensor is None else tensor
        if not tensor.is_contiguous():
            raise ValueError(f"native checkpoint requires a contiguous shard: {key}")
        if tensor.numel() != self.tensor.numel():
            raise ValueError(f"optimizer and model shard lengths differ: {key}")
        # Uneven original-parameter boundaries are supported by Core's native
        # CheckpointableShardedTensor; no flattened_range or fake rank axis.
        data = tensor.detach().view(-1)
        return ShardedTensor(
            key,
            data,
            data.dtype,
            tuple(data.shape),
            (math.prod(self.shape),),
            (self.offset,),
            axis_fragmentations=None,
            replica_id=self.replica,
        )


def model_slices(backend, state_filter):
    """Select draft state, retaining the real ownership of every local view."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    module = backend.module
    if module is None:
        raise RuntimeError("native checkpoint requires a prepared model")
    if backend._wrapper_kind == "ddp":
        module = module.module
    rank = dist.get_rank()
    managed, ignored = {}, set()
    if backend._wrapper_kind == "fsdp":
        from torch.distributed.fsdp._runtime_utils import _lazy_init

        # Mirror FSDP's own pre-state-dict initialization. It casts buffers and
        # initializes local shard attributes without gathering model weights.
        # A step-0 checkpoint then has the same buffer semantics as first forward.
        _lazy_init(module, module)
        for fsdp in FSDP.fsdp_modules(module):
            if not fsdp._use_orig_params:
                raise ValueError("native checkpoint requires FSDP use_orig_params=True")
            ignored.update(fsdp._ignored_params)
            handle = fsdp._handle
            if handle is None:
                continue
            if handle.world_size != dist.get_world_size() or handle.rank != rank:
                raise ValueError("native checkpoint requires FSDP on the WORLD group")
            flat = handle.flat_param
            for param, shape, info in zip(
                flat._params, flat._shapes, flat._shard_param_infos, strict=True
            ):
                count = info.numel_in_shard if info.in_shard else 0
                if param.numel() != count:
                    raise ValueError(
                        "FSDP parameters must be sharded at an optimizer boundary; "
                        f"expected {count} elements, found {param.numel()}"
                    )
                managed[id(param)] = TensorSlice(
                    param,
                    tuple(shape),
                    info.intra_param_start_idx if info.in_shard else 0,
                    0,
                )
    candidates = {}
    for name, param in module.named_parameters(remove_duplicate=False):
        view = managed.get(id(param))
        if view is None:
            if backend._wrapper_kind == "fsdp" and param not in ignored:
                raise ValueError(f"missing FSDP ownership metadata for {name}")
            view = TensorSlice(param, tuple(param.shape), 0, rank)
        candidates[_clean_name(name)] = view
    for prefix, child in module.named_modules(remove_duplicate=False):
        for name, buffer in child._buffers.items():
            if buffer is not None and name not in child._non_persistent_buffers_set:
                key = _clean_name(f"{prefix}.{name}".lstrip("."))
                candidates[key] = TensorSlice(
                    buffer, tuple(buffer.shape), 0, rank, (child, name)
                )
    selected = state_filter(candidates)
    if not selected or any(not isinstance(v, TensorSlice) for v in selected.values()):
        raise ValueError("checkpoint state filter must select named draft tensor views")
    return selected


class NativeState:
    """One checkpoint's live tensor descriptors and small rank-local sidecar."""

    def __init__(self, backend, state_filter):
        self.backend = backend
        self.model = model_slices(backend, state_filter)
        opt = backend.optimizer
        if not isinstance(opt, BF16Optimizer):
            raise TypeError("native FSDP checkpoints require BF16Optimizer")
        self.opt = opt
        by_id = {}
        for name, view in sorted(self.model.items()):
            by_id.setdefault(id(view.tensor), name)
        self.names = []
        for param in opt.model_params:
            if id(param) not in by_id:
                raise ValueError(
                    "optimizer owns a parameter excluded by checkpoint policy"
                )
            self.names.append(by_id[id(param)])

    def layout(self):
        slots = []
        for master in self.opt.fp32_params:
            state = self.opt.optimizer.state.get(master, {})
            fields = sorted(set(state) - {"step"})
            if set(fields) - {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
                raise ValueError(f"unsupported Adam tensor fields: {fields}")
            slots.append(fields)
        return {
            "model": {name: view.layout() for name, view in sorted(self.model.items())},
            "optimizer_names": self.names,
            "optimizer_slots": slots,
        }

    def build(self, training, layouts, *, restoring=False):
        from nexflashcheckpoint import object_shard

        rank, world = dist.get_rank(), dist.get_world_size()
        layout = layouts[rank]
        current = self.layout()
        if restoring:
            # Adam is lazy in a newly built optimizer; saved field presence
            # determines its load targets. Parameter ownership must match exactly.
            current["optimizer_slots"] = layout["optimizer_slots"]
            for name, view in self.model.items():
                if view.buffer_owner is not None and name in layout["model"]:
                    current["model"][name]["dtype"] = layout["model"][name]["dtype"]
        if current != layout:
            raise ValueError("native checkpoint FSDP/model/optimizer layout mismatch")
        if restoring:
            # FSDP casts floating buffers on first forward. Restore the saved
            # runtime dtype even when this target has not run its first forward.
            converted = {}
            for name, view in self.model.items():
                if view.buffer_owner is not None:
                    dtype = getattr(
                        torch, layout["model"][name]["dtype"].removeprefix("torch.")
                    )
                    key = (id(view.tensor), dtype)
                    if key not in converted:
                        converted[key] = view.tensor.to(dtype=dtype)
                    view.tensor = converted[key]
                    owner, buffer_name = view.buffer_owner
                    owner._buffers[buffer_name] = view.tensor
        tensors = {}
        for name, view in self.model.items():
            if view.tensor.numel():
                tensors[f"model/{name}"] = view.descriptor(f"model/{name}")
        inner = self.opt.optimizer.state_dict()
        sidecar = {
            "optimizer_state_dict": {
                "state": {},
                "param_groups": copy.deepcopy(inner["param_groups"]),
            },
            "scheduler_state_dict": copy.deepcopy(self.opt.scheduler.state_dict()),
            "lr_scheduler_type": self.opt.lr_scheduler_type,
            "max_grad_norm": self.opt.max_grad_norm,
            "rng": self.backend._rng_state(),
        }
        for index, (name, master, fields) in enumerate(
            zip(
                self.names, self.opt.fp32_params, layout["optimizer_slots"], strict=True
            )
        ):
            view = self.model[name]
            state = self.opt.optimizer.state.get(master, {})
            # Only small per-parameter step counters enter the atomic sidecar.
            if "step" in state:
                sidecar["optimizer_state_dict"]["state"][index] = {
                    "step": state["step"].detach().cpu().clone()
                }
            if not master.numel():
                continue
            key = f"optimizer/master/{name}"
            tensors[key] = view.descriptor(key, master)
            for field in fields:
                value = torch.empty_like(master) if restoring else state[field]
                if value.dtype != torch.float32 or value.shape != master.shape:
                    raise ValueError(f"Adam state shape/dtype mismatch: {name}/{field}")
                key = f"optimizer/{field}/{name}"
                tensors[key] = view.descriptor(key, value)
        return {
            "training": training,
            "nfc_layout": layouts,
            "tensors": tensors,
            "rank_state": object_shard(
                "trainer/rank_state",
                sidecar,
                global_shape=(world,),
                global_offset=(rank,),
            ),
        }

    def apply(self, loaded, layout):
        tensors, sidecar = loaded["tensors"], loaded["rank_state"]
        with torch.no_grad():
            for name, view in self.model.items():
                if view.tensor.numel():
                    view.tensor.copy_(tensors[f"model/{name}"].view_as(view.tensor))
        optimizer = {k: v for k, v in sidecar.items() if k != "rng"}
        optimizer["fp32_params"] = []
        for index, (name, master, fields) in enumerate(
            zip(
                self.names, self.opt.fp32_params, layout["optimizer_slots"], strict=True
            )
        ):
            optimizer["fp32_params"].append(
                tensors[f"optimizer/master/{name}"].view_as(master)
                if master.numel()
                else master
            )
            for field in fields:
                optimizer["optimizer_state_dict"]["state"].setdefault(index, {})[
                    field
                ] = (
                    tensors[f"optimizer/{field}/{name}"].view_as(master)
                    if master.numel()
                    else torch.empty_like(master)
                )
        self.opt.load_state_dict(optimizer)
        self.backend._set_rng_state(sidecar["rng"])
