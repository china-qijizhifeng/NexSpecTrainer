"""torchrun --nproc_per_node=8 ... --output PATH --phase write|resume.

Actual nested FSDP1, gradient accumulation/clipping, empty/uneven shards,
FP32/BF16 NS, offload and new-process native NFC resume. Supply --reference
with the pinned TorchTitan dist_muon.py to run its actual numerical functions.
"""

import argparse
import ast
import copy
import hashlib
import json
import os
import typing
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn

from specforge.config.schema import TrainingConfig
from specforge.muon import MuonAdamWOptimizer, orthogonalize
from specforge.muon_config import V41_FLASH_DEFAULTS, MuonConfig
from specforge.muon_layout import capture_layout
from specforge.training.assembly import _ConfiguredOptimizerFactory
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.checkpoint import CheckpointManager
from specforge.training.nfc_state import model_slices


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(29, 29)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return x + self.dropout(torch.tanh(self.linear(x)))


class Draft(nn.Module):
    _no_split_modules = ("Block",)

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(17, 29)
        self.blocks = nn.Sequential(Block(), Block())
        self.norm = nn.LayerNorm(29)
        self.scalar = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.norm(self.blocks(self.proj(x))) * self.scalar


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.draft_model = Draft().bfloat16()
        self.lm_head = nn.Linear(29, 17).bfloat16().requires_grad_(False)

    def forward(self, x):
        return self.lm_head(self.draft_model(x))


def state_filter(state):
    return {
        k.removeprefix("draft_model."): v
        for k, v in state.items()
        if k.startswith("draft_model.")
    }


def make(dtype, offload):
    torch.manual_seed(42)
    model = Model().cuda()
    config = MuonConfig(**V41_FLASH_DEFAULTS, ns_dtype=dtype)
    training = TrainingConfig(
        optimizer="muon_adamw",
        muon=config,
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
    backend.prepare_model(model, optimizer_target=model.draft_model)
    return backend


def snapshot(backend):
    return copy.deepcopy(
        {
            "model": {
                k: v.tensor.detach().cpu().clone()
                for k, v in model_slices(backend, state_filter).items()
            },
            "optimizer": backend.optimizer.state_dict(),
            "rng": backend._rng_state(),
        }
    )


def equal(a, b, path="root"):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu()), (a.shape, (a.cpu() - b.cpu()).abs().max())
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            equal(a[k], b[k], path + "." + str(k))
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    elif hasattr(a, "shape"):
        assert (a == b).all()
    else:
        assert a == b, (a, b)


def gradients(backend):
    losses = []
    for micro in range(3):
        inputs = torch.randn(4, 17, device="cuda", dtype=torch.bfloat16)
        loss = backend.module(inputs).float().square().mean() / 3
        backend.backward(loss, is_boundary=micro == 2)
        losses.append(loss.detach())
    return torch.stack(losses)


def update(backend):
    losses = gradients(backend)
    norm = backend.step()
    assert torch.isfinite(norm) and norm > 0.1
    return losses


def differential(backend):
    # Compare the exact FSDP-reduced gradients to an unsharded optimizer. This
    # isolates optimizer arithmetic from different forward/reduction rounding.
    reference = Draft().bfloat16().cuda()
    options = backend.optimizer.muon_config
    plan = capture_layout(reference, options)
    opt = MuonAdamWOptimizer(
        reference,
        plan=plan,
        config=options,
        lr=0.003,
        total_steps=20,
        warmup_ratio=0.1,
        max_grad_norm=0.1,
        offload_master=backend.optimizer.offload_master,
    )
    opt.configure_grad_norm_reduction(enabled=False)
    max_error = 0.0
    for step in range(3):
        gradients(backend)
        for item, master, refp, refmaster in zip(
            backend.optimizer.plan,
            backend.optimizer.fp32_params,
            reference.parameters(),
            opt.fp32_params,
        ):
            full = torch.zeros(item.shape, device="cuda", dtype=torch.float32)
            if master.numel():
                full.view(-1)[item.offset : item.offset + master.numel()].copy_(master)
            dist.all_reduce(full)
            refmaster.data.copy_(full)
            refp.data.copy_(full)
            grad = torch.zeros_like(full)
            if item.parameter.grad is not None:
                grad.view(-1)[item.offset : item.offset + master.numel()].copy_(
                    item.parameter.grad
                )
            dist.all_reduce(grad)
            refp.grad = grad.to(refp.dtype)
        actual_norm = backend.step()
        expected_norm = opt.step()
        tolerance = 2e-6 if options.ns_dtype == "float32" else 2e-4
        for item, master, refmaster in zip(
            backend.optimizer.plan, backend.optimizer.fp32_params, opt.fp32_params
        ):
            expected = refmaster.view(-1)[
                item.offset : item.offset + master.numel()
            ].view_as(master)
            if not torch.allclose(master, expected, atol=tolerance, rtol=tolerance):
                torch.save(
                    {
                        "name": item.name,
                        "actual": master.detach().cpu(),
                        "expected": expected.detach().cpu(),
                        "actual_norm": actual_norm.cpu(),
                        "expected_norm": expected_norm.cpu(),
                    },
                    Path(os.environ.get("RUN_ROOT", "/tmp"))
                    / f"differential-failure-rank{dist.get_rank()}.pt",
                )
            torch.testing.assert_close(master, expected, atol=tolerance, rtol=tolerance)
            if master.numel():
                max_error = max(max_error, (master - expected).abs().max().item())
    return max_error


def reference_check(path):
    # Execute just the real, pinned pure numerical helpers. Future annotations
    # avoid importing TorchTitan's unrelated FlexShard dependencies into FSDP1.
    source = path.read_text()
    expected = "035040ac524e98f6f7ca5f3b249fee054152bc84"
    blob = source.encode()
    assert (
        hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest()
        == expected
    )
    names = {
        "_is_ns_coefficient_triple",
        "_ns_coefficients_by_step",
        "_zeropower_via_newtonschulz",
        "_prepare_muon_input",
    }
    tree = ast.parse(source)
    body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    body += [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    env = {
        "torch": torch,
        "cast": typing.cast,
        "_NSCoefficients": tuple,
        "_NSCoefficientSchedule": tuple,
    }
    exec(compile(module, str(path), "exec"), env)
    maximum = 0.0
    for shape in [(7, 19), (19, 7), (9, 9)]:
        for nesterov in (True, False):
            config = MuonConfig(
                nesterov=nesterov,
                ns_steps=10,
                ns_coefficients=V41_FLASH_DEFAULTS["ns_coefficients"],
            )
            ours = torch.zeros(shape, device="cuda")
            theirs = ours.clone()
            for step in range(4):
                grad = torch.randn_like(ours) if step != 2 else torch.zeros_like(ours)
                ours.lerp_(grad, 1 - config.momentum)
                prepared = (
                    torch.lerp(grad, ours, config.momentum)
                    if nesterov
                    else ours.clone()
                )
                expected_input = env["_prepare_muon_input"](
                    grad,
                    theirs,
                    momentum=config.momentum,
                    nesterov=nesterov,
                    out=torch.empty_like(grad),
                )
                assert torch.equal(prepared, expected_input)
                expected_update = env["_zeropower_via_newtonschulz"](
                    expected_input,
                    ns_coefficients=config.ns_coefficients,
                    ns_steps=10,
                    eps=config.eps,
                )
                actual = orthogonalize(prepared, config)
                assert torch.equal(actual, expected_update)
                maximum = max(maximum, (actual - expected_update).abs().max().item())
    return maximum


def run(args):
    rank = dist.get_rank()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "rank": rank,
        "phase": args.phase,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "cases": [],
    }
    if args.reference:
        result["torchtitan_ns_max_error"] = reference_check(args.reference)
        (args.output / f"torchtitan-rank{rank}.json").write_text(
            json.dumps(result) + "\n"
        )
    for dtype, offload in [("float32", False), ("bfloat16", True)]:
        name = f"{dtype}-offload{int(offload)}"
        root = args.output / name
        backend = make(dtype, offload)
        frozen = backend.module.lm_head.weight.detach().clone()
        manager = CheckpointManager(
            str(root),
            "muon",
            nfc_options={
                "max_shared_memory_bytes": 16 << 20,
                "pipeline_max_inflight_bytes": 16 << 20,
                "staging_tile_bytes": 1 << 20,
                "load_worker_count": 2,
            },
        )
        try:
            if args.phase == "write":
                error = differential(backend)
                (args.output / f"differential-{name}-rank{rank}.json").write_text(
                    json.dumps(
                        {
                            "passed": True,
                            "max_abs_error": error,
                            "ns_dtype": dtype,
                            "offload": offload,
                            "tf32": torch.backends.cuda.matmul.allow_tf32,
                        }
                    )
                    + "\n"
                )
                gradients(backend)
                before_invalid = snapshot(backend)
                if rank == 0:
                    next(
                        p
                        for p in backend.optimizer.model_params
                        if p.numel() and p.grad is not None
                    ).grad.view(-1)[0] = float("nan")
                try:
                    backend.step()
                except FloatingPointError:
                    pass
                else:
                    raise AssertionError("nonfinite rank must fail on all ranks")
                equal(snapshot(backend), before_invalid)
                # Fresh lazy state, independent of the reference comparison.
                backend = make(dtype, offload)
                for step in range(3):
                    if step:
                        update(backend)
                    before = snapshot(backend)
                    path = manager.save_native(
                        backend,
                        state_filter,
                        {
                            "global_step": step,
                            "world_size": dist.get_world_size(),
                            "run_id": "muon",
                        },
                        step,
                    )
                    manager.finalize_pending()
                    losses = update(backend)
                    expected = snapshot(backend)
                    torch.save(
                        {"before": before, "after": expected, "losses": losses.cpu()},
                        root / f"expected-step{step}-rank{rank}.pt",
                    )
                    manager.restore_native(backend, state_filter, path)
                    equal(snapshot(backend), before)
                result["cases"].append(
                    {
                        "case": name,
                        "reference_max_abs_error": error,
                        "empty_shards": sum(
                            p.parameter.numel() == 0 for p in backend.optimizer.plan
                        ),
                        "lazy_and_initialized_saved": True,
                    }
                )
            else:
                for step in range(3):
                    fresh = make(dtype, offload)
                    path = manager.checkpoint_dir(step)
                    expected = torch.load(
                        root / f"expected-step{step}-rank{rank}.pt", weights_only=False
                    )
                    manager.restore_native(fresh, state_filter, path)
                    equal(snapshot(fresh), expected["before"])
                    losses = update(fresh)
                    equal(losses.cpu(), expected["losses"])
                    equal(snapshot(fresh), expected["after"])
                result["cases"].append(
                    {
                        "case": name,
                        "fresh_process_next_step_bitwise": True,
                        "lazy_and_initialized_restored": True,
                    }
                )
            assert torch.equal(backend.module.lm_head.weight, frozen)
        finally:
            manager.close()
    result["passed"] = True
    (args.output / f"{args.phase}-rank{rank}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["write", "resume"], required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    from specforge.distributed import _distributed_backend

    dist.init_process_group(_distributed_backend("cuda"))
    try:
        run(args)
    finally:
        dist.destroy_process_group()
