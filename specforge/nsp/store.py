"""Durable run receipts. Observation never imports the training runtime."""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import uuid
from pathlib import Path

TERMINAL = {"succeeded", "failed", "cancelled"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class RunStore:
    def __init__(self, root=None):
        self.root = (
            Path(root or os.environ.get("NSP_ROOT", "~/.local/share/nsp"))
            .expanduser()
            .resolve()
        )

    def path(self, run_id):
        if not re.fullmatch(r"(?:tr|ev|ca)_[a-f0-9]{24}", run_id):
            raise ValueError(f"invalid run ID: {run_id!r}")
        return self.root / "runs" / run_id

    @contextlib.contextmanager
    def lock(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.root / ".lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def create(self, kind, name, config, request_id):
        fingerprint = digest({"kind": kind, "name": name, "config": config})
        # The caller holds lock through create + launch receipt publication.
        for run in self.list():
            if run["request_id"] == request_id:
                if run["fingerprint"] != fingerprint:
                    raise ValueError(
                        "request ID already belongs to a different submission"
                    )
                return run["id"], False
        run_id = {"train": "tr_", "eval": "ev_", "calibrate": "ca_"}[kind] + uuid.uuid4().hex[:24]
        path = self.path(run_id)
        path.mkdir(parents=True, mode=0o700)
        atomic_json(
            path / "run.json",
            {
                "schema_version": 1,
                "id": run_id,
                "kind": kind,
                "name": name,
                "request_id": request_id,
                "fingerprint": fingerprint,
                "created_at": now(),
                "attempt": 1,
            },
        )
        atomic_json(path / "intent.json", config)
        return run_id, True

    def attempt(self, run_id):
        run = read_json(self.path(run_id) / "run.json")
        return self.path(run_id) / "attempts" / f'{run["attempt"]:04d}'

    def status(self, run_id):
        run = read_json(self.path(run_id) / "run.json")
        attempt = self.attempt(run_id)
        status = (
            read_json(attempt / "status.json")
            if (attempt / "status.json").exists()
            else {"state": "created"}
        )
        result = {**run, **status, "path": str(self.path(run_id))}
        if status.get("heartbeat_at") and status["state"] not in TERMINAL:
            age = (
                dt.datetime.now(dt.timezone.utc)
                - dt.datetime.fromisoformat(status["heartbeat_at"])
            ).total_seconds()
            result["observation"] = "stale" if age > 15 else "current"
        return result

    def list(self):
        runs = self.root / "runs"
        if not runs.exists():
            return []
        return sorted(
            (self.status(p.name) for p in runs.iterdir() if (p / "run.json").is_file()),
            key=lambda r: r["created_at"],
            reverse=True,
        )


def checkpoint_entries(output_dir):
    """Mirror CheckpointManager's completed publication contract without torch."""
    entries = []
    for path in Path(output_dir).glob("*-step*"):
        match = re.search(r"-step(\d+)$", path.name)
        if not match or path.is_symlink():
            continue
        complete = (path / "training_state.pt").is_file() or (
            (path / "training_state.json").is_file()
            and all(
                (path / "nfc" / name).is_file()
                for name in ("common.pt", "metadata.json", ".metadata")
            )
        )
        if complete:
            entries.append(
                {"step": int(match[1]), "path": str(path.resolve()), "complete": True}
            )
    return sorted(entries, key=lambda e: e["step"])


def resolve_checkpoint(store, reference):
    if "@" not in reference:
        path = Path(reference.removeprefix("file://")).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"checkpoint does not exist: {path}")
        return str(path)
    run_id, selector = reference.split("@", 1)
    match = re.fullmatch(r"step-?(\d+)", selector)
    for attempt in sorted((store.path(run_id) / "attempts").iterdir(), reverse=True):
        intent = read_json(attempt / "resolved.json")
        if "output_dir" not in intent:
            continue
        entries = checkpoint_entries(intent["output_dir"])
        if selector == "latest" and entries:
            return entries[-1]["path"]
        if match:
            for entry in entries:
                if entry["step"] == int(match[1]):
                    return entry["path"]
    raise ValueError(f"no completed checkpoint for {reference}")


@contextlib.contextmanager
def checkpoint_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".nsp-checkpoint.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def snapshot_checkpoint(source, destination):
    """Keep published immutable shards alive independently of training rotation."""
    import shutil

    source, destination = Path(source).resolve(), Path(destination)
    if source.is_file() and source.name == "training_state.json":
        source = source.parent

    def link_or_copy(src, dst):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    pretrained = (source / "config.json").is_file() and any(
        (source / name).is_file()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )
    # A read-only exported model is not subject to CheckpointManager rotation.
    lock = contextlib.nullcontext() if pretrained else checkpoint_lock(source.parent)
    with lock:
        if source.is_file():
            if source.name not in {"training_state.pt", "training_state.json"}:
                raise ValueError(
                    "pass a completed checkpoint directory or training_state.pt"
                )
            destination.mkdir()
            link_or_copy(source, destination / source.name)
            return str(destination / source.name)
        native = (source / "training_state.pt").is_file() or (
            (source / "training_state.json").is_file()
            and all(
                (source / "nfc" / name).is_file()
                for name in ("common.pt", "metadata.json", ".metadata")
            )
        )
        if not (native or pretrained):
            raise ValueError(
                f"checkpoint is incomplete or not a model directory: {source}"
            )
        shutil.copytree(source, destination, copy_function=link_or_copy)
    return str(destination)
