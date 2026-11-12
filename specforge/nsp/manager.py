"""Resolve and freeze work before starting its independent supervisor."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from specforge.nsp.store import (
    TERMINAL,
    atomic_json,
    now,
    read_json,
    resolve_checkpoint,
)


def source_snapshot(destination):
    import specforge

    source = Path(specforge.__file__).parent
    shutil.copytree(
        source,
        destination / "specforge",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    hasher = hashlib.sha256()
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            hasher.update(str(path.relative_to(destination)).encode())
            hasher.update(path.read_bytes())
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True
    )
    return {
        "commit": result.stdout.strip() if result.returncode == 0 else None,
        "source_sha256": hasher.hexdigest(),
        "python": sys.executable,
    }


def train_config(path, overrides):
    from specforge.application import resolve_run
    from specforge.config import load_config
    from specforge.launch_plan import build_launch_plan

    cfg = load_config(path, overrides)
    if (
        cfg.deployment.trainer.nnodes != 1
        or "RANK" in os.environ
        or "LOCAL_RANK" in os.environ
        or int(os.environ.get("WORLD_SIZE", "1")) != 1
    ):
        raise ValueError(
            "nsp submissions must run outside torchrun and currently support one trainer node"
        )
    if cfg.training.role == "producer":
        raise ValueError(
            "nsp train manages a complete training run, not a standalone producer"
        )
    resolved = resolve_run(cfg)
    launch = build_launch_plan(
        cfg,
        algorithm=resolved.algorithm,
        config_path=str(Path(path).resolve()),
        requested_role="auto",
        node_rank=None,
        overrides=[],
    )
    raw = cfg.model_dump(mode="json")
    # Preserve the invocation's path interpretation when workers detach.
    for section in ("model", "data"):
        for key, value in raw[section].items():
            if (
                isinstance(value, str)
                and value
                and (
                    key.endswith("path")
                    or key.endswith("dir")
                    or key == "draft_model_config"
                )
            ):
                if Path(value).exists():
                    raw[section][key] = str(Path(value).resolve())
    return raw, launch


def submit(store, kind, config, *, name=None, request_id=None, resume_run=None):
    request_id = request_id or str(uuid.uuid4())
    with store.lock():
        if resume_run:
            run_id = resume_run
            run = read_json(store.path(run_id) / "run.json")
            if store.status(run_id)["state"] not in TERMINAL:
                raise ValueError("resume requires a terminal attempt")
            run["attempt"] += 1
            atomic_json(store.path(run_id) / "run.json", run)
            created = True
        else:
            run_id, created = store.create(kind, name, config, request_id)
        if not created:
            return store.status(run_id)
        attempt = store.attempt(run_id)
        attempt.mkdir(parents=True)
        effective = json.loads(json.dumps(config))
        if kind == "train":
            effective["run_id"] = run_id
            effective["output_dir"] = str(attempt / "checkpoints")
        atomic_json(attempt / "resolved.json", effective)
        try:
            if kind == "eval":
                from specforge.nsp.store import snapshot_checkpoint

                offline = any(m not in ("al", "performance") for m in effective["metrics"])
                if offline:
                    frozen = snapshot_checkpoint(effective["checkpoint"], attempt / "checkpoint")
                else:
                    # A serving eval never loads weights. Cross-fileset hard
                    # links can fall back to copying a multi-hundred-GB model;
                    # preserve its external metadata identity instead.
                    from specforge.nsp.eval_inputs import serving_checkpoint_identity

                    atomic_json(attempt / "checkpoint-identity.json",
                                serving_checkpoint_identity(effective["checkpoint"]))
                if effective.get("resolved_recipe"):
                    effective["resolved_recipe"]["model"][
                        "draft_checkpoint_path"
                    ] = frozen
                if effective.get("serving"):
                    prompts = attempt / "prompts.jsonl"
                    prompts.write_bytes(Path(effective["serving"]["prompts"]).read_bytes())
                    effective["serving"]["prompts"] = str(prompts)
            elif kind == "train" and effective["training"].get("resume_from"):
                from specforge.nsp.store import snapshot_checkpoint

                effective["training"]["resume_from"] = snapshot_checkpoint(
                    effective["training"]["resume_from"], attempt / "resume-checkpoint"
                )
            if kind == "calibrate":
                original = Path(effective["traces"])
                data = original.read_bytes()
                frozen = attempt / "traces.jsonl"
                frozen.write_bytes(data)
                effective["traces"] = str(frozen)
                effective["provenance"]["traces_sha256"] = hashlib.sha256(data).hexdigest()
            atomic_json(attempt / "resolved.json", effective)
            provenance = source_snapshot(attempt / "source")
            argv = [sys.executable, "-m", "specforge.nsp.worker", str(attempt), kind]
            manifest = {
                "schema_version": 1,
                "kind": kind,
                "created_at": now(),
                "cwd": str(Path.cwd()),
                "argv": argv,
                **provenance,
            }
            atomic_json(attempt / "manifest.json", manifest)
            atomic_json(
                attempt / "status.json", {"state": "created", "heartbeat_at": now()}
            )
            env = os.environ.copy()
            env["PYTHONSAFEPATH"] = "1"
            env["PYTHONPATH"] = (
                str(attempt / "source") + os.pathsep + env.get("PYTHONPATH", "")
            )
            command = [sys.executable, "-m", "specforge.nsp.supervisor", str(attempt)]
            with (attempt / "supervisor.log").open("ab") as log:
                supervisor = subprocess.Popen(
                    command,
                    env=env,
                    cwd="/",
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            atomic_json(
                attempt / "submission.json",
                {"pid": supervisor.pid, "at": now(), "request_id": request_id},
            )
        except BaseException as exc:
            atomic_json(
                attempt / "status.json",
                {"state": "failed", "error": str(exc), "finished_at": now()},
            )
            raise
    return store.status(run_id)


def resume_config(store, reference):
    run_id = reference.split("@", 1)[0]
    previous = read_json(store.attempt(run_id) / "resolved.json")
    if store.status(run_id)["kind"] != "train":
        raise ValueError("only training runs can resume")
    previous["training"]["resume_from"] = resolve_checkpoint(store, reference)
    previous["model"]["draft_checkpoint_path"] = None
    from specforge.application import resolve_run
    from specforge.config import Config

    resolve_run(Config.model_validate(previous))
    return run_id, previous
