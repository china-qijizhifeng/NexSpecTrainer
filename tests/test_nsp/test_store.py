from pathlib import Path

import pytest

from specforge.nsp.store import RunStore, checkpoint_entries, snapshot_checkpoint


def test_submission_retry_is_idempotent_and_conflicting_reuse_fails(tmp_path):
    store = RunStore(tmp_path)
    with store.lock():
        first, created = store.create("train", "run", {"x": 1}, "request-1")
        second, again = store.create("train", "run", {"x": 1}, "request-1")
        assert first == second and created and not again
        with pytest.raises(ValueError, match="different submission"):
            store.create("train", "run", {"x": 2}, "request-1")
    assert len(store.list()) == 1
    with pytest.raises(ValueError):
        store.path("../../other")


def test_checkpoint_snapshot_survives_rotation(tmp_path):
    import shutil

    source = tmp_path / "output" / "run-step10"
    source.mkdir(parents=True)
    (source / "training_state.pt").write_bytes(b"checkpoint")
    incomplete = tmp_path / "output" / "run-step11"
    incomplete.mkdir()
    assert [r["step"] for r in checkpoint_entries(source.parent)] == [10]
    frozen = Path(snapshot_checkpoint(source, tmp_path / "frozen"))
    shutil.rmtree(source)
    assert (frozen / "training_state.pt").read_bytes() == b"checkpoint"
    with pytest.raises(ValueError, match="incomplete"):
        snapshot_checkpoint(incomplete, tmp_path / "bad")
