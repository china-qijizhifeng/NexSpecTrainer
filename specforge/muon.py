# Newton–Schulz arithmetic adapted from TorchTitan (Meta Platforms, Inc.).
# See docs/licenses/torchtitan.txt for its BSD license.
"""Muon + AdamW over FP32 masters, streaming complete FSDP logical matrices.

Algorithm reference: pytorch/torchtitan a97b83bbbfebb9bf121641890f48e9986f7f3c78,
DistMuon. This adapter targets FSDP1, not TorchTitan's DTensor/FlexShard runtime.
"""

import math

import torch
import torch.distributed as dist

from specforge.optimizer import BF16Optimizer

IMPLEMENTATION = "specforge-muon-fsdp1.v1"
REFERENCE = "pytorch/torchtitan@a97b83bbbfebb9bf121641890f48e9986f7f3c78"


def orthogonalize(matrix, config):
    """Quintic NS with TorchTitan's normalization and addmm arithmetic."""
    x = matrix.to(dtype=getattr(torch, config.ns_dtype), copy=True)
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x.div_(x.norm().clamp_min(config.eps))
    coefficients = config.ns_coefficients
    if isinstance(coefficients[0], (int, float)):
        coefficients = (coefficients,) * config.ns_steps
    for a, b, c in coefficients:
        gram = x @ x.T
        polynomial = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.addmm(x, polynomial, x, beta=a)
    return x.T if transpose else x


def muon_direction(matrix, config):
    direction = orthogonalize(matrix, config).float()
    rows, cols = matrix.shape
    if config.scaling == "original":
        direction.mul_(math.sqrt(max(1, rows / cols)))
    elif config.scaling == "match_rms_adamw":
        direction.mul_(0.2 * math.sqrt(max(rows, cols)))
    elif config.scaling == "v41_rms":
        direction.mul_(config.rms * math.sqrt(max(rows, cols)))
    else:
        # Actual per-matrix RMS, not the approximate spectral-size multiplier.
        direction.mul_(
            config.rms / direction.square().mean().sqrt().clamp_min(config.eps)
        )
    return direction


class _HybridOptimizer(torch.optim.Optimizer):
    def __init__(self, masters, plan, config, lr):
        self.plan, self.config = plan, config
        groups = []
        for master, item in zip(masters, plan, strict=True):
            muon = item.matrix_shape is not None
            weight_decay = (
                config.muon_weight_decay if muon else config.adamw_weight_decay
            )
            if config.profile == "v41_flash_muon" and item.role in {
                "bias_or_scale",
                "markov",
            }:
                weight_decay = 0.0
            groups.append(
                {
                    "params": [master],
                    "algorithm": "muon" if muon else "adamw",
                    "lr": lr
                    * (
                        config.muon_lr_multiplier
                        if muon
                        else config.adamw_lr_multiplier
                    ),
                    "weight_decay": weight_decay,
                }
            )
        super().__init__(groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Muon closure steps are unsupported")
        config = self.config
        device = self.plan[0].parameter.device
        active = torch.tensor(
            [int(g["params"][0].grad is not None) for g in self.param_groups],
            device=device,
        )
        # Empty FSDP shards participate in exactly the same collective sequence.
        if any(item.sharded for item in self.plan):
            dist.all_reduce(
                active,
                op=dist.ReduceOp.MAX,
                group=self.plan[
                    next(i for i, p in enumerate(self.plan) if p.sharded)
                ].process_group,
            )
        active = active.tolist()
        for enabled, group, item in zip(
            active, self.param_groups, self.plan, strict=True
        ):
            if not enabled:
                continue
            master = group["params"][0]
            grad = master.grad if master.grad is not None else torch.zeros_like(master)
            lr, wd = group["lr"], group["weight_decay"]
            if group["algorithm"] == "adamw":
                if not master.numel():
                    continue
                state = self.state[master]
                if not state:
                    state.update(
                        step=torch.tensor(0.0),
                        exp_avg=torch.zeros_like(master),
                        exp_avg_sq=torch.zeros_like(master),
                    )
                state["step"].add_(1)
                beta1, beta2 = config.adamw_betas
                state["exp_avg"].lerp_(grad, 1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                step = int(state["step"].item())
                denominator = (
                    state["exp_avg_sq"]
                    .sqrt()
                    .div_(math.sqrt(1 - beta2**step))
                    .add_(config.adamw_eps)
                )
                master.mul_(1 - lr * wd).addcdiv_(
                    state["exp_avg"], denominator, value=-lr / (1 - beta1**step)
                )
                continue
            state = self.state[master]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(master)
            momentum = state["momentum_buffer"]
            if config.momentum_convention == "sum":
                momentum.mul_(config.momentum).add_(grad)
            else:
                momentum.lerp_(grad, 1 - config.momentum)
            size = math.prod(item.matrix_shape)
            shard_start, shard_end = item.offset, item.offset + master.numel()
            # One matrix at a time, including each expert and each Q head. No
            # full expert stack, flat FSDP buffer, or whole-model gather exists.
            for start in range(0, math.prod(item.shape), size):
                left, right = max(start, shard_start), min(start + size, shard_end)
                matrix = torch.zeros(
                    item.matrix_shape, dtype=torch.float32, device=item.parameter.device
                )
                if left < right:
                    local = slice(left - shard_start, right - shard_start)
                    prepared = momentum.view(-1)[local]
                    if config.nesterov:
                        if config.momentum_convention == "sum":
                            prepared = grad.view(-1)[local].add(
                                prepared, alpha=config.momentum
                            )
                        else:
                            prepared = torch.lerp(
                                grad.view(-1)[local], prepared, config.momentum
                            )
                    matrix.view(-1)[left - start : right - start].copy_(prepared)
                if item.sharded:
                    dist.all_reduce(matrix, group=item.process_group)
                update = muon_direction(matrix, config)
                if left < right:
                    local_master = master.view(-1)[local]
                    local_master.mul_(1 - lr * wd).add_(
                        update.reshape(-1)[left - start : right - start].to(
                            master.device
                        ),
                        alpha=-lr,
                    )
                del matrix, update


class MuonAdamWOptimizer(BF16Optimizer):
    """Share the existing accumulation, global clipping and master-copy path."""

    def __init__(self, model, *, plan, config, **kwargs):
        self.plan, self.muon_config = plan, config
        if [id(p) for p in model.parameters() if p.requires_grad] != [
            id(p.parameter) for p in plan
        ]:
            raise ValueError("Muon parameter order changed during wrapping")
        super().__init__(model, **kwargs)
        self.optimizer_identity = {
            "implementation": IMPLEMENTATION,
            "reference": REFERENCE,
            "config": config.model_dump(mode="json"),
            "parameters": [item.identity() for item in plan],
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "base_lr": kwargs["lr"],
            "lr_scheduler": self.lr_scheduler_type,
            "max_grad_norm": self.max_grad_norm,
            "total_steps": kwargs.get("total_steps", 800_000),
            "warmup_ratio": kwargs.get("warmup_ratio", 0.015),
        }

    def _build_optimizer(self, lr, weight_decay):
        return _HybridOptimizer(self.fp32_params, self.plan, self.muon_config, lr)

    def _restore_adamw_implementation(self):
        # AdamW counters are scalar CPU tensors; moment/master placement follows
        # the current run. No fused-Adam flags belong to the hybrid optimizer.
        for state in self.optimizer.state.values():
            if "step" in state:
                state["step"] = state["step"].cpu()

    def load_state_dict(self, state_dict):
        self.validate_checkpoint_identity(state_dict.get("optimizer_identity"))
        masters = state_dict.get("fp32_params")
        if masters is None or len(masters) != len(self.fp32_params):
            raise ValueError("Muon resume requires all FP32 masters")
        for saved, master in zip(masters, self.fp32_params, strict=True):
            if saved.shape != master.shape or saved.dtype != torch.float32:
                raise ValueError("Muon master shape/dtype mismatch")
        saved_groups = state_dict["optimizer_state_dict"]["param_groups"]
        if len(saved_groups) != len(self.optimizer.param_groups):
            raise ValueError("Muon optimizer group count mismatch")
        for saved, current in zip(
            saved_groups, self.optimizer.param_groups, strict=True
        ):
            if any(saved[key] != current[key] for key in ("algorithm", "weight_decay")):
                raise ValueError("Muon optimizer group algorithm/decay mismatch")
        for index, state in state_dict["optimizer_state_dict"]["state"].items():
            allowed = self.allowed_state_fields(index)
            fields = set(state) - {"step"}
            if fields != allowed:
                raise ValueError("Muon optimizer state schema mismatch")
            for field in fields:
                value = state[field]
                if (
                    value.shape != self.fp32_params[index].shape
                    or value.dtype != torch.float32
                ):
                    raise ValueError("Muon optimizer state shape/dtype mismatch")
        super().load_state_dict(state_dict)

    def allowed_state_fields(self, index):
        return (
            {"momentum_buffer"}
            if self.plan[index].matrix_shape
            else {"exp_avg", "exp_avg_sq"}
        )
