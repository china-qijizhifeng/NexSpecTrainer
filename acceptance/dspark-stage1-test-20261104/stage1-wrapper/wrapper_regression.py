"""Execute the exact temporary forward auditor with the real native strategy."""

import ast
import hashlib
import json
import tempfile
import time
from pathlib import Path

import torch

from specforge.algorithms.deepseek_v41.providers import build_step
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import DSparkTrainStrategy
from tests.test_modeling.test_dspark_anchor_padding import build_model


def extract(path, name, namespace):
    tree = ast.parse(path.read_text())
    owner = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "run_training"
    )
    node = next(
        n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def check(path, expect_failure):
    model = build_model()
    strategy = build_step(model)
    assert not hasattr(strategy, "model")
    original = DSparkTrainStrategy.forward_loss
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        consumed, eval_count = [], [0]
        namespace = dict(
            root=root,
            rank=0,
            consumed=consumed,
            eval_count=eval_count,
            hashlib=hashlib,
            json=json,
            time=time,
            torch=torch,
            forward_loss=original,
        )
        audited = extract(path, "audited_loss", namespace)
        batch = TrainBatch(
            sample_ids=["sparse-real-strategy"],
            strategy="deepseek_v41",
            tensors={
                "input_ids": torch.randint(0, 30, (1, 20)),
                "loss_mask": torch.ones(1, 20),
                "hidden_states": torch.randn(1, 20, 16),
                "target_last_hidden_states": torch.randn(1, 20, 16),
            },
        )
        if expect_failure:
            try:
                audited(strategy, batch)
            except AttributeError as error:
                assert "model" in str(error)
            else:
                raise AssertionError("old audit did not reproduce")
            return
        result = audited(strategy, batch)
        result.loss.backward()
        assert len(consumed) == 1 and eval_count == [0]
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        )
        model.eval()
        with torch.no_grad():
            audited(strategy, batch)
        assert len(consumed) == 1 and eval_count == [1]
        model.train()
        audited(strategy, batch)
        assert len(consumed) == 2 and eval_count == [1]
        assert (
            json.loads((root / "micro-progress-rank0.json").read_text())["micro_step"]
            == 2
        )
        assert (
            json.loads((root / "eval-progress-rank0.json").read_text())["micro_step"]
            == 1
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-audit", type=Path)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    check(
        args.previous_audit
        or base.parent / "dspark-denovo-20261104-stage1-r5/audit_train.py",
        True,
    )
    check(base / "audit_train.py", False)
    print(
        json.dumps(
            {
                "passed": True,
                "old_failure_reproduced": True,
                "real_native_strategy": True,
                "train_eval_train_calls": 3,
                "train_micro_count": 2,
                "eval_micro_count": 1,
                "finite_backward": True,
            }
        )
    )
