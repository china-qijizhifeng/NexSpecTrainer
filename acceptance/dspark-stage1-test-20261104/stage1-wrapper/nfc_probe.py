"""Bounded hashes around the planned first-step native save and fresh restore."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from specforge.training.nfc_checkpoint import NativeCheckpoint, read_manifest
from specforge.training.nfc_state import model_slices


def state_hash(backend, state_filter):
    digest = hashlib.sha256()

    def visit(value):
        digest.update(type(value).__qualname__.encode())
        if isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            flat = value.detach().reshape(-1)
            for start in range(0, flat.numel(), 1 << 20):
                digest.update(
                    flat[start : start + (1 << 20)]
                    .cpu()
                    .contiguous()
                    .view(torch.uint8)
                    .numpy()
                )
        elif isinstance(value, np.ndarray):
            digest.update(str((value.dtype, value.shape)).encode())
            digest.update(value.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=repr):
                visit(key)
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())
        digest.update(b"\0")

    visit(
        {
            "model": {
                name: view.tensor
                for name, view in model_slices(backend, state_filter).items()
            },
            "optimizer": backend.optimizer.state_dict(),
            "rng": backend._rng_state(),
        }
    )
    return digest.hexdigest()


def install_probe(root, rank, phase):
    state = {"restored": False}
    before_path = Path(root) / f"nfc-step1-state-rank{rank}.json"
    save, restore = NativeCheckpoint.save, NativeCheckpoint.restore

    def checked_save(self, backend, state_filter, training, step):
        if phase == "initial" and step == 1:
            assert (
                training["epoch_samples"] == 16
                and training["effective_total_steps"] == 118
            )
            before_path.write_text(
                json.dumps(
                    {
                        "rank": rank,
                        "sha256": state_hash(backend, state_filter),
                        "training": training,
                    }
                )
                + "\n"
            )
        return save(self, backend, state_filter, training, step)

    def checked_restore(self, backend, state_filter, path):
        manifest = read_manifest(path)
        assert manifest["training"]["global_step"] == 1
        restore(self, backend, state_filter, path)
        expected = json.loads(before_path.read_text())
        # JSON evidence turns tuples into lists; native common.pt preserves
        # the original typed contract. Compare their canonical JSON forms.
        assert json.dumps(manifest["training"], sort_keys=True) == json.dumps(
            expected["training"], sort_keys=True
        )
        actual = state_hash(backend, state_filter)
        assert (
            actual == expected["sha256"]
        ), "full model/masters/Adam/scheduler/RNG restore differs"
        state.update(
            restored=True, sha256=actual, checkpoint=str(path), restored_step=1
        )
        (Path(root) / f"full-model-restored-rank{rank}.json").write_text(
            json.dumps(state) + "\n"
        )

    NativeCheckpoint.save, NativeCheckpoint.restore = checked_save, checked_restore
    return state
