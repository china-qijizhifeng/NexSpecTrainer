"""Ownership and lifecycle gates for the native FSDP checkpoint integration."""

import hashlib
import json
import os
import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from specforge.config.schema import NFCCheckpointConfig
from specforge.training.checkpoint import CheckpointManager
from specforge.training.nfc_checkpoint import validate_layouts
from specforge.training.nfc_state import FORMAT, MANIFEST, TensorSlice, model_slices


def _layouts():
    return [
        {
            "model": {
                "w": {
                    "shape": [7],
                    "offset": offset,
                    "numel": count,
                    "dtype": "torch.float32",
                    "replica": 0,
                }
            },
            "optimizer_names": ["w"],
            "optimizer_slots": [["exp_avg", "exp_avg_sq"]],
        }
        for offset, count in [(0, 4), (4, 3), (0, 0)]
    ]


def test_uneven_and_empty_parameter_shards():
    validate_layouts(_layouts())


@pytest.mark.parametrize("mutation", ["overlap", "hole", "dtype", "names", "moments"])
def test_invalid_ownership_is_rejected(mutation):
    layouts = _layouts()
    if mutation == "overlap":
        layouts[1]["model"]["w"]["offset"] = 3
    elif mutation == "hole":
        layouts[1]["model"]["w"]["numel"] = 2
    elif mutation == "dtype":
        layouts[1]["model"]["w"]["dtype"] = "torch.bfloat16"
    elif mutation == "names":
        layouts[1]["optimizer_names"] = ["other"]
    else:
        layouts[1]["optimizer_slots"] = [[]]
    with pytest.raises(ValueError):
        validate_layouts(layouts)


def test_replicas_are_counted_once():
    layouts = _layouts()
    for rank, layout in enumerate(layouts):
        layout["model"]["w"].update(offset=0, numel=7, replica=rank)
    validate_layouts(layouts)


def test_options_cannot_enable_generic_or_nonpipeline_saving():
    assert NFCCheckpointConfig().pipeline_max_inflight_bytes == 2 << 30
    for options in [
        {"pipeline": False},
        {"backend": "torch_dcp"},
        {"max_shared_memory_bytes": 0},
        {"writer_thread_count": True},
    ]:
        with pytest.raises(ValueError):
            NFCCheckpointConfig.model_validate(options)


def test_model_views_keep_aliases_buffers_and_policy_without_state_dict():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.draft_model = torch.nn.Linear(3, 2)
            self.draft_model.register_buffer("indices", torch.arange(2))
            self.draft_model.register_buffer(
                "temporary", torch.arange(2), persistent=False
            )
            self.target = torch.nn.Linear(3, 2)

        def state_dict(self, *args, **kwargs):
            raise AssertionError("native save must not gather a full state dict")

    model = Model()
    backend = SimpleNamespace(module=model, _wrapper_kind="none")
    with mock.patch("torch.distributed.get_rank", return_value=0):
        views = model_slices(
            backend,
            lambda state: {
                k.removeprefix("draft_model."): v
                for k, v in state.items()
                if k.startswith("draft_model.")
            },
        )
    assert set(views) == {"weight", "bias", "indices"}
    assert views["weight"].tensor is model.draft_model.weight
    assert views["weight"].shape == (2, 3)
    assert isinstance(views["indices"], TensorSlice)


def test_native_manifest_is_visible_only_after_dcp_publication(tmp_path):
    manager = CheckpointManager(str(tmp_path), "run", nfc_options={})
    path = Path(manager.checkpoint_dir(2))
    (path / "nfc").mkdir(parents=True)
    training = {
        "global_step": 2,
        "world_size": 1,
        "strategy": "dspark",
        "run_id": "run",
        "specforge_step_options": (("fused", True),),
        "nested": {"layers": (1, 2)},
    }
    common_path = path / "nfc" / "common.pt"
    torch.save({"training": training, "nfc_layout": [{}]}, common_path)
    marker = {
        "format": FORMAT,
        "global_step": 2,
        "run_id": "run",
        "common_sha256": hashlib.sha256(common_path.read_bytes()).hexdigest(),
    }
    (path / MANIFEST).write_text(json.dumps(marker))
    assert manager.latest_dir() is None
    for name in ("metadata.json", ".metadata"):
        (path / "nfc" / name).touch()
    assert manager.latest_dir() == str(path)
    with mock.patch("torch.load", wraps=torch.load) as load:
        state = manager.read_resume_state(str(tmp_path))
    assert load.call_count == 1
    assert str(load.call_args.args[0]).endswith("nfc/common.pt")
    assert state["specforge_step_options"] == training["specforge_step_options"]
    assert state["nested"] == training["nested"]
    assert state["_nfc_checkpoint"] == str(path)
    assert state["global_step"] == 2
    assert state["backend"] == {}
    from specforge.launch import _checkpoint_global_step

    # Online resume reads this on rank 0 before the collective Trainer setup.
    with mock.patch(
        "torch.distributed.all_gather_object",
        side_effect=AssertionError("rank-0 metadata read must not be collective"),
    ):
        assert _checkpoint_global_step(str(tmp_path)) == 2
        assert _checkpoint_global_step(f"file://{path / MANIFEST}") == 2


def test_native_writer_failure_does_not_publish_latest(tmp_path):
    from specforge.training.nfc_checkpoint import NativeCheckpoint

    manager = CheckpointManager(str(tmp_path), "run", nfc_options={})
    writer = NativeCheckpoint(manager, {})
    fake_state = SimpleNamespace(layout=lambda: _layouts()[0], build=lambda *args: {})
    ticket = SimpleNamespace(wait=mock.Mock(side_effect=RuntimeError("disk failed")))
    runtime = SimpleNamespace(save=mock.Mock(return_value=ticket))
    with (
        mock.patch(
            "specforge.training.nfc_checkpoint.NativeState", return_value=fake_state
        ),
        mock.patch("specforge.training.nfc_checkpoint.validate_layouts"),
        mock.patch("torch.distributed.get_world_size", return_value=1),
        mock.patch("torch.distributed.broadcast_object_list"),
        mock.patch(
            "torch.distributed.all_gather_object",
            side_effect=lambda out, value: out.__setitem__(0, value),
        ),
        mock.patch.object(writer, "_runtime", return_value=runtime),
        pytest.raises(RuntimeError, match="disk failed"),
    ):
        writer.save(None, None, {"world_size": 1}, 1)
    assert manager.latest_dir() is None
    assert not (Path(manager.checkpoint_dir(1)) / MANIFEST).exists()


@pytest.fixture
def async_writer(tmp_path):
    """Control local capture and write completion at the NFC ticket boundary."""
    from specforge.training.nfc_checkpoint import NativeCheckpoint

    manager = CheckpointManager(str(tmp_path), "run", nfc_options={}, max_checkpoints=1)
    writer = NativeCheckpoint(manager, {})
    manager._nfc = writer
    tickets = []

    def save(state, path, **kwargs):
        path = Path(path)
        path.mkdir()
        torch.save(state, path / "common.pt")
        for name in ("metadata.json", ".metadata"):
            (path / name).touch()
        capture, durable = Future(), Future()
        capture.set_result(None)
        ticket = SimpleNamespace(
            capture=capture,
            durable=durable,
            wait=mock.Mock(
                side_effect=lambda milestone, timeout=None: (
                    capture if milestone == "source_released" else durable
                ).result(timeout)
            ),
        )
        tickets.append(ticket)
        return ticket

    writer.runtime = SimpleNamespace(save=save, close=mock.Mock())
    state = SimpleNamespace(
        layout=lambda: {},
        build=lambda training, layouts: {"training": training, "nfc_layout": layouts},
    )

    def submit(step):
        return writer.save(
            None, None, {"world_size": 1, "run_id": "run", "global_step": step}, step
        )

    with (
        mock.patch("specforge.training.nfc_checkpoint.NativeState", return_value=state),
        mock.patch("specforge.training.nfc_checkpoint.validate_layouts"),
        mock.patch("torch.distributed.get_world_size", return_value=1),
        mock.patch("torch.distributed.broadcast_object_list"),
        mock.patch(
            "torch.distributed.all_gather_object",
            side_effect=lambda out, value: out.__setitem__(0, value),
        ),
    ):
        yield manager, writer, tickets, submit


def test_async_source_fence_defers_best_latest_and_rotation(async_writer):
    manager, writer, tickets, submit = async_writer
    first = submit(1)
    tickets[-1].durable.set_result(None)
    manager.update_best(1, {manager.best_metric: 1.0})
    assert manager.finalize_pending()
    second = submit(2)
    assert tickets[-1].wait.call_args == mock.call("source_released")
    manager.update_best(2, {manager.best_metric: 3.0})
    assert not manager.finalize_pending(wait=False)
    assert manager.latest_dir() == first
    assert manager.best_step == 1
    assert Path(first).is_dir()
    assert not Path(second).exists()
    assert not manager.is_better({manager.best_metric: 2.0})
    assert manager.is_better({manager.best_metric: 4.0})
    tickets[-1].durable.set_result(None)
    assert manager.finalize_pending(wait=False)
    assert manager.latest_dir() == second
    assert manager.best_step == 2
    assert not Path(first).exists()
    assert writer.pending is None


@pytest.mark.parametrize("step", [1, 2])
def test_async_write_failure_preserves_previous_complete_best(async_writer, step):
    manager, writer, tickets, submit = async_writer
    previous = submit(1)
    tickets[-1].durable.set_result(None)
    manager.update_best(1, {manager.best_metric: 1.0})
    manager.finalize_pending()
    old_marker = (Path(previous) / MANIFEST).read_bytes()
    submit(step)
    manager.update_best(step, {manager.best_metric: 2.0})
    tickets[-1].durable.set_exception(OSError("disk full"))
    with pytest.raises(OSError, match="disk full"):
        manager.finalize_pending(wait=False)
    assert manager.latest_dir() == previous
    assert manager.best_step == 1
    assert (Path(previous) / MANIFEST).read_bytes() == old_marker
    assert writer.pending is not None


def test_next_save_and_close_drain_before_reusing_pool(async_writer):
    manager, writer, tickets, submit = async_writer
    submit(1)
    first = tickets[-1]
    first.durable.set_result(None)
    submit(2)
    assert first.wait.call_args == mock.call("recoverable", timeout=None)
    assert manager.latest_dir() == manager.checkpoint_dir(1)
    runtime = writer.runtime
    tickets[-1].durable.set_result(None)
    manager.close()
    runtime.close.assert_called_once()
    assert writer.pending is None
    assert manager.latest_dir() == manager.checkpoint_dir(2)


def test_pending_rollback_compares_against_new_timeline_best(async_writer):
    manager, _, tickets, submit = async_writer
    submit(5)
    tickets[-1].durable.set_result(None)
    manager.update_best(5, {manager.best_metric: 10.0})
    manager.finalize_pending()
    submit(2)
    manager.update_best(2, {manager.best_metric: 2.0})
    assert manager.is_better({manager.best_metric: 3.0})
    assert not manager.is_better({manager.best_metric: 1.0})
    assert manager.best_step == 5  # old complete version survives until commit


def test_plain_save_syncs_install_before_deleting_old_versions(async_writer):
    manager, writer, tickets, submit = async_writer
    submit(2)
    tickets[-1].durable.set_result(None)
    events = []
    final = manager.checkpoint_dir(2)
    directory_inode = os.stat(manager.output_dir).st_ino
    replace, fsync, rewind, rotate = (
        os.replace,
        os.fsync,
        manager._rewind,
        manager._rotate,
    )

    def recorded_replace(src, dst):
        if dst == final:
            events.append("install")
        replace(src, dst)

    def recorded_fsync(fd):
        if os.fstat(fd).st_ino == directory_inode:
            events.append("sync-output")
        fsync(fd)

    def recorded_rewind(*args, **kwargs):
        events.append("rewind")
        rewind(*args, **kwargs)

    def recorded_rotate(*args, **kwargs):
        events.append("rotate")
        rotate(*args, **kwargs)

    with (
        mock.patch("os.replace", side_effect=recorded_replace),
        mock.patch("os.fsync", side_effect=recorded_fsync),
        mock.patch.object(manager, "_rewind", side_effect=recorded_rewind),
        mock.patch.object(manager, "_rotate", side_effect=recorded_rotate),
    ):
        writer.finalize_pending()
    assert events == ["install", "sync-output", "rewind", "rotate", "sync-output"]


def test_published_directory_uses_the_checkpoint_umask(async_writer):
    manager, _, tickets, submit = async_writer
    previous_umask = os.umask(0o027)
    try:
        path = submit(1)
        tickets[-1].durable.set_result(None)
        manager.finalize_pending()
    finally:
        os.umask(previous_umask)
    assert os.stat(path).st_mode & 0o777 == 0o750


def test_native_runtime_always_caches_plans_and_defers_capture(async_writer):
    _, writer, _, _ = async_writer
    writer.runtime = None
    create = mock.Mock()
    with mock.patch.dict(
        sys.modules,
        {"nexflashcheckpoint": SimpleNamespace(MegatronCheckpointRuntime=create)},
    ):
        writer._runtime()
    assert create.call_args.kwargs["cached_metadata"] is True
    assert create.call_args.kwargs["defer_source_release"] is True


@pytest.mark.parametrize(
    "saved_dtype,target_dtype",
    [(torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)],
)
def test_restore_preserves_saved_buffer_dtype_across_first_forward(
    saved_dtype, target_dtype
):
    from specforge.optimizer import BF16Optimizer
    from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
    from specforge.training.nfc_state import NativeState

    model = torch.nn.Linear(3, 2).bfloat16()
    model.register_buffer("runtime_buffer", torch.ones(2, dtype=saved_dtype))
    model.register_buffer("empty", torch.empty(0, dtype=saved_dtype))
    backend = FSDPTrainingBackend(
        ParallelConfig(), optimizer_factory=lambda m: BF16Optimizer(m, lr=0.01)
    )
    backend.prepare_model(model, wrap=False)
    mapping = SimpleNamespace(
        ShardedTensor=lambda *args, **kwargs: SimpleNamespace(data=args[1])
    )
    nfc = SimpleNamespace(object_shard=lambda key, data, **kwargs: data)
    with (
        mock.patch("torch.distributed.get_rank", return_value=0),
        mock.patch("torch.distributed.get_world_size", return_value=1),
        mock.patch.dict(
            sys.modules,
            {
                "nexflashcheckpoint": nfc,
                "megatron.core.dist_checkpointing.mapping": mapping,
            },
        ),
    ):
        layouts = [NativeState(backend, lambda x: x).layout()]
        model.runtime_buffer = model.runtime_buffer.to(target_dtype)
        model.empty = model.empty.to(target_dtype)
        target = NativeState(backend, lambda x: x)
        target.build({"world_size": 1}, layouts, restoring=True)
        assert model.runtime_buffer.dtype == saved_dtype
        assert model.empty.dtype == saved_dtype
        model.weight.data = model.weight.data.float()
        with pytest.raises(ValueError, match="layout mismatch"):
            NativeState(backend, lambda x: x).build(
                {"world_size": 1}, layouts, restoring=True
            )


def test_export_rejects_missing_declared_weights(tmp_path):
    from torch.distributed.checkpoint import FileSystemWriter, save_state_dict

    from specforge.training.nfc_checkpoint import load_draft_weights

    payload = {"model/good": torch.arange(3.0)}
    save_state_dict(payload, FileSystemWriter(str(tmp_path / "nfc")), no_dist=True)
    item = {"shape": [3], "dtype": "torch.float32"}
    common_path = tmp_path / "nfc" / "common.pt"
    torch.save(
        {
            "training": {"world_size": 1, "global_step": 1, "run_id": "test"},
            "nfc_layout": [{"model": {"good": item, "missing": item}}],
        },
        common_path,
    )
    manifest = {
        "format": FORMAT,
        "global_step": 1,
        "run_id": "test",
        "common_sha256": hashlib.sha256(common_path.read_bytes()).hexdigest(),
    }
    (tmp_path / MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(BaseException, match="Missing key.*model/missing"):
        load_draft_weights(str(tmp_path))
