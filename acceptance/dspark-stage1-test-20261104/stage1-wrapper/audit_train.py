"""Audit the public training CLI; safe when NFC spawn imports __mp_main__."""


def spawn_preflight():
    import json
    import multiprocessing as mp
    import sys
    from contextlib import ExitStack

    # Exact context and Manager/Queue construction used by Core's async writer.
    # Keep all eight managers live together, one for each trainer rank.
    with ExitStack() as stack:
        queues = []
        for rank in range(8):
            manager = stack.enter_context(mp.get_context("spawn").Manager())
            queue = manager.Queue()
            queue.put(rank)
            queues.append(queue)
        assert [queue.get(timeout=10) for queue in queues] == list(range(8))
    assert "torch" not in sys.modules and "specforge.cli" not in sys.modules
    print(
        json.dumps(
            {
                "passed": True,
                "spawn_managers": 8,
                "queues_round_tripped": 8,
                "training_imported": False,
            }
        ),
        flush=True,
    )
    return 0


def run_training():
    import hashlib
    import importlib.metadata as md
    import json
    import os
    import sys
    import time
    from pathlib import Path

    import torch

    from specforge.training.backend import FSDPTrainingBackend
    from specforge.training.strategies.base import DSparkTrainStrategy

    root = Path(
        "/gpfs/users/linlizhi/dspark-denovo-specforge-stage1-20261104-r6/training"
    )
    root.mkdir(exist_ok=True)
    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 8
    assert torch.__version__ == "2.10.0a0+b558c986e8.nv25.11"
    assert md.version("transformer-engine") == "2.20.2+6ea2a74"
    observed = {}
    phase = os.environ["DSPARK_TRAIN_PHASE"]
    assert phase in ("initial", "resume")
    start_step = 0 if phase == "initial" else 1
    expected_steps = 1 if phase == "initial" else 118
    prefix = root.parent / f"initial-consumed-rank{rank}.json"
    consumed = [] if phase == "initial" else json.loads(prefix.read_text())
    initial_norms = (
        []
        if phase == "initial"
        else json.loads((root.parent / f"initial-audit-rank{rank}.json").read_text())[
            "grad_norms"
        ]
    )
    eval_count = [0]
    assert len(consumed) == 16 * start_step
    from nfc_probe import install_probe

    probe = install_probe(root.parent, rank, phase)
    forward_loss = DSparkTrainStrategy.forward_loss

    def audited_loss(self, batch, ctx=None):
        ids = batch.tensors["input_ids"].detach().cpu().contiguous()
        is_training = self.trainable_module().training
        if is_training:
            consumed.append(hashlib.sha256(ids.view(torch.uint8).numpy()).hexdigest())
        else:
            eval_count[0] += 1
        mask = batch.tensors["loss_mask"].detach().cpu()
        anchors = ((mask[:, :-1] > 0.5) & (mask[:, 1:] > 0.5)).sum(-1).tolist()
        kind = "micro-progress" if is_training else "eval-progress"
        micro = {
            "rank": rank,
            "micro_step": len(consumed) if is_training else eval_count[0],
            "time": time.time(),
            "valid_anchors": anchors,
            "phase": "forward",
            "training": is_training,
        }
        tmp = root / f".{kind}-rank{rank}.partial"
        tmp.write_text(json.dumps(micro) + "\n")
        tmp.replace(root / f"{kind}-rank{rank}.json")
        result = forward_loss(self, batch, ctx)
        micro.update(phase="forward_complete", time=time.time())
        tmp.write_text(json.dumps(micro) + "\n")
        tmp.replace(root / f"{kind}-rank{rank}.json")
        assert torch.isfinite(result.loss).all().item()
        return result

    DSparkTrainStrategy.forward_loss = audited_loss
    prepare = FSDPTrainingBackend.prepare_model
    step = FSDPTrainingBackend.step

    def checksum(parameter):
        value = parameter.detach().reshape(-1)
        digest = hashlib.sha256()
        for start in range(0, value.numel(), 4 << 20):
            digest.update(
                value[start : start + (4 << 20)].cpu().view(torch.uint8).numpy()
            )
        return digest.hexdigest()

    def audited_prepare(self, model, **options):
        wrapped = prepare(self, model, **options)
        assert not observed, "expected one training backend"
        parameters = [p for m in self.ignored_frozen_modules for p in m.parameters()]
        assert len(parameters) == 2
        assert all(not p.requires_grad and p.grad is None for p in parameters)
        assert {id(p) for p in parameters}.isdisjoint(
            id(p) for p in self.optimizer.model_params
        )
        observed.update(
            backend=self,
            parameters=parameters,
            before=[checksum(p) for p in parameters],
            steps=[],
        )
        (root / f"frozen-baseline-rank{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "hashes": observed["before"],
                    "excluded_from_optimizer": True,
                    "requires_grad": False,
                    "gradients_absent": True,
                }
            )
            + "\n"
        )
        return wrapped

    def audited_step(self, **options):
        assert observed["backend"] is self
        assert all(
            not p.requires_grad and p.grad is None for p in observed["parameters"]
        )
        norm = step(self, **options)
        assert (
            norm is not None and torch.isfinite(norm).all().item() and float(norm) > 0
        )
        observed["steps"].append(float(norm))
        completed = start_step + len(observed["steps"])
        progress = {
            "rank": rank,
            "optimizer_steps": completed,
            "grad_norms": initial_norms + observed["steps"],
            "checkpoint_complete": False,
        }
        temporary = root / f".live-progress-rank{rank}.partial"
        temporary.write_text(json.dumps(progress, indent=2) + "\n")
        temporary.replace(root / f"live-progress-rank{rank}.json")
        if phase == "resume" and completed == 2:
            assert probe["restored"]
            (root.parent / f"first-restored-update-rank{rank}.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "rank": rank,
                        "restored_step": 1,
                        "next_optimizer_step": 2,
                        "grad_norm": float(norm),
                        "full_state_sha256": probe["sha256"],
                        "checkpoint": probe["checkpoint"],
                        "micro_steps_consumed": len(consumed),
                    }
                )
                + "\n"
            )
        return norm

    FSDPTrainingBackend.prepare_model = audited_prepare
    FSDPTrainingBackend.step = audited_step
    from specforge.cli import main

    code = main(sys.argv[1:])
    assert code == 0, code
    assert start_step + len(observed["steps"]) == expected_steps, observed["steps"]
    after = [checksum(p) for p in observed["parameters"]]
    assert (
        after == observed["before"]
    ), "frozen target tables changed during actual updates"
    assert all(not p.requires_grad and p.grad is None for p in observed["parameters"])
    result = {
        "passed": True,
        "rank": rank,
        "torch": torch.__version__,
        "te": md.version("transformer-engine"),
        "optimizer_steps": expected_steps,
        "grad_norms": initial_norms + observed["steps"],
        "consumed_input_sha256": consumed,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "frozen_table_sha256_before": observed["before"],
        "frozen_table_sha256_after": after,
        "frozen_parameters_excluded_from_optimizer": True,
        "frozen_gradients_absent": True,
        "scope": "full frozen target-table hashes around the complete stage-one epoch on this rank",
    }
    if phase == "initial":
        prefix.write_text(json.dumps(consumed) + "\n")
        (root.parent / f"initial-audit-rank{rank}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    else:
        assert probe["restored"] and len(consumed) == 118 * 16
        (root / f"live-audit-rank{rank}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--spawn-preflight"]:
        raise SystemExit(spawn_preflight())
    raise SystemExit(run_training())
