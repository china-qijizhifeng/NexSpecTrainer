# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""On-disk checkpoint lifecycle: layout, rotation, best tracking, resume reads.

A checkpoint is ``{run_id}-step{N}/`` under ``output_dir``. Native NFC saves
hold ``nfc/`` shards and a ``training_state.json`` completion marker. Legacy
saves hold ``training_state.pt``
(rank0 shared payload) plus ``training_state_rank{r}.pt`` per rank. FSDP optimizer
and all RNG state are rank-local; replicated DDP optimizer state is stored once
in the shared payload. Run-scoped ``{run_id}-latest``/``{run_id}-best`` links and
``{run_id}.best_meta.json`` sit beside them; multi-rank runs need a shared fs.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import shutil
from typing import Any, Dict, Iterator, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

STATE_FILE = "training_state.pt"


class CheckpointManager:
    def __init__(
        self,
        output_dir: str,
        run_id: str,
        *,
        max_checkpoints: int = 0,
        best_metric: str = "eval/simulated_acc_len",
        best_min_delta: float = 0.0,
        nfc_options: Optional[dict] = None,
    ) -> None:
        self.output_dir = output_dir
        self.run_id = run_id
        self.max_checkpoints = max_checkpoints
        self.best_metric = best_metric
        self.best_min_delta = best_min_delta
        self.native = nfc_options is not None
        self._nfc = None
        self._nfc_options = nfc_options
        self.best_score: Optional[float] = None
        self.best_step: Optional[int] = None
        self._load_best_meta()

    def _native_checkpoint(self):
        if not self.native:
            raise ValueError(
                "set training.nfc_checkpoint to use native checkpoint restore"
            )
        if self._nfc is None:
            from specforge.training.nfc_checkpoint import NativeCheckpoint

            self._nfc = NativeCheckpoint(self, self._nfc_options)
        return self._nfc

    def save_native(self, backend, state_filter, training, step):
        return self._native_checkpoint().save(backend, state_filter, training, step)

    def restore_native(self, backend, state_filter, path):
        return self._native_checkpoint().restore(backend, state_filter, path)

    def finalize_pending(self, *, wait=True):
        """Collectively publish a ready native save; wait only at drain points."""
        return self._nfc is None or self._nfc.finalize_pending(wait=wait)

    def close(self):
        if self._nfc is not None:
            self._nfc.close()

    @staticmethod
    def is_complete(path):
        from specforge.training.nfc_state import MANIFEST

        return os.path.isfile(os.path.join(path, STATE_FILE)) or (
            os.path.isfile(os.path.join(path, MANIFEST))
            and all(
                os.path.isfile(os.path.join(path, "nfc", name))
                for name in ("common.pt", "metadata.json", ".metadata")
            )
        )

    # -- layout ------------------------------------------------------------
    def checkpoint_dir(self, step: int) -> str:
        return os.path.join(self.output_dir, f"{self.run_id}-step{step}")

    def _state_path(self, ckpt_dir: str) -> str:
        return os.path.join(ckpt_dir, STATE_FILE)

    @staticmethod
    def _rank_file(rank: int) -> str:
        return f"training_state_rank{rank}.pt"

    @property
    def _best_meta_path(self) -> str:
        return os.path.join(self.output_dir, f"{self.run_id}.best_meta.json")

    # -- save --------------------------------------------------------------
    def save(
        self,
        state: Optional[Dict[str, Any]],
        step: int,
        *,
        rank_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write ``step``'s checkpoint, first deleting any on-disk steps >= step
        (fork/rollback semantics), then repoint ``{run_id}-latest`` and rotate.
        ``state`` is the shared payload (rank0 writes it); ``rank_state`` is
        written by every rank. Collective: any rank's failure raises on all ranks.
        """
        ckpt_dir = self.checkpoint_dir(step)
        err = ""
        try:
            if self.is_rank0():
                self._rewind(step)
        except Exception as exc:
            err = f"rewind failed: {type(exc).__name__}: {exc}"
        self._barrier()  # stale >= step dirs are gone before any rank recreates them
        if not err:
            try:
                os.makedirs(ckpt_dir, exist_ok=True)
                if rank_state is not None:
                    self._atomic_save(
                        rank_state,
                        os.path.join(ckpt_dir, self._rank_file(self._rank())),
                    )
                if self.is_rank0() and state is not None:
                    self._atomic_save(state, self._state_path(ckpt_dir))
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
        self._all_ok(err)
        if self.is_rank0():
            self._point(f"{self.run_id}-latest", ckpt_dir)
            self._rotate(keep_step=step)
        self._barrier()  # no rank proceeds before the checkpoint is complete
        return ckpt_dir

    def _rewind(self, step: int, *, keep_path: Optional[str] = None) -> None:
        from specforge.nsp.store import checkpoint_lock

        with checkpoint_lock(self.output_dir):
            self._rewind_unlocked(step, keep_path=keep_path)

    def _rewind_unlocked(self, step: int, *, keep_path: Optional[str] = None) -> None:
        # Fork semantics: saving step S invalidates on-disk steps >= S, including
        # a best record inside the deleted range.
        stale = sorted(
            path for s, path in self._step_dirs() if s >= step and path != keep_path
        )
        if stale:
            logger.warning(
                "rewinding checkpoint timeline at step %d: deleting %s",
                step,
                ", ".join(stale),
            )
            for path in stale:
                shutil.rmtree(path)
        if self.best_step is not None and self.best_step >= step:
            self.best_score = None
            self.best_step = None
            best_link = os.path.join(self.output_dir, f"{self.run_id}-best")
            for path in (self._best_meta_path, best_link):
                try:
                    os.remove(path)
                except OSError:
                    pass  # best-effort: reload ignores meta whose step dir is gone

    def _all_ok(self, err: str) -> None:
        # Barrier replacement: every rank learns every rank's outcome, so one
        # rank's FS failure raises everywhere instead of stranding the group.
        if torch.distributed.is_initialized():
            errs = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(errs, err)
        else:
            errs = [err]
        failures = [(r, e) for r, e in enumerate(errs) if e]
        if failures:
            raise RuntimeError(
                "checkpoint save failed: "
                + "; ".join(f"rank {r}: {e}" for r, e in failures)
            )

    # -- best tracking -----------------------------------------------------
    def score(self, eval_metrics: Optional[Dict[str, Any]]) -> Optional[float]:
        """Best-tracking score from an eval-metrics dict; None when absent."""
        s = (eval_metrics or {}).get(self.best_metric)
        return float(s) if s is not None else None

    def is_better(self, eval_metrics: Optional[Dict[str, Any]]) -> bool:
        """Rank-agreed verdict: do these metrics beat the tracked best (higher
        wins, by more than ``best_min_delta``)? False for empty metrics.
        Collective: rank0 decides and broadcasts, so every rank must call it.
        """
        verdict = False
        if self.is_rank0():
            s = self.score(eval_metrics)
            best = self.best_score
            if self._nfc is not None and self._nfc.pending is not None:
                if (
                    self.best_step is not None
                    and self.best_step >= self._nfc.pending.step
                ):
                    best = None  # publication will rewind this future best
                candidate = self.score(self._nfc.pending.best_metrics)
                if candidate is not None:
                    best = candidate if best is None else max(best, candidate)
            verdict = s is not None and (best is None or s > best + self.best_min_delta)
        return self._bcast_flag(verdict)

    def update_best(self, step: int, eval_metrics: Dict[str, Any]) -> None:
        """Record ``step`` as the tracked best unconditionally (callers gate with
        ``is_better``); rank0 repoints ``{run_id}-best`` and persists the meta."""
        if self._nfc is not None and self._nfc.pending is not None:
            if self._nfc.pending.step != step:
                raise ValueError("best step does not match the in-flight checkpoint")
            self._nfc.pending.best_metrics = dict(eval_metrics)
            return
        self.best_score = self.score(eval_metrics)
        self.best_step = step
        if not self.is_rank0():
            return
        self._point(f"{self.run_id}-best", self.checkpoint_dir(step))
        self._atomic_json(
            {
                "run_id": self.run_id,
                "step": step,
                "score": self.best_score,
                "metric": self.best_metric,
                "metrics": eval_metrics,
            },
            self._best_meta_path,
        )

    def _load_best_meta(self) -> None:
        # Rehydrate so a restart neither rotates away the on-disk best nor lets
        # a worse score overwrite it.
        try:
            with open(self._best_meta_path) as fh:
                meta = json.load(fh)
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            logger.warning(
                "ignoring unreadable best meta %s: %s", self._best_meta_path, exc
            )
            return
        step = meta.get("step")
        score = meta.get("score")
        if (
            meta.get("run_id") != self.run_id
            or meta.get("metric") != self.best_metric
            or not isinstance(step, int)
            or not isinstance(score, (int, float))
            or not self.is_complete(self.checkpoint_dir(step))
        ):
            logger.warning(
                "ignoring best meta %s: run_id/metric mismatch, malformed, or its "
                "checkpoint is gone",
                self._best_meta_path,
            )
            return
        self.best_score, self.best_step = float(score), int(step)

    # -- load --------------------------------------------------------------
    def load(self, step: Optional[int] = None, *, map_location="cpu") -> Dict[str, Any]:
        """Load a checkpoint's shared state dict (``latest`` when ``step`` is None)."""
        target = self.checkpoint_dir(step) if step is not None else self.latest_dir()
        if target is None:
            raise FileNotFoundError(f"no checkpoint to load under {self.output_dir}")
        if os.path.isfile(os.path.join(target, "training_state.json")):
            from specforge.training.nfc_checkpoint import load_draft_weights

            return load_draft_weights(target)
        return torch.load(
            self._state_path(target), map_location=map_location, weights_only=False
        )

    @classmethod
    def read_resume_state(
        cls,
        path: str,
        *,
        map_location="cpu",
        require_full_state: bool = True,
    ) -> Dict[str, Any]:
        """Read resume state. Native checkpoints return typed common metadata and
        ``_nfc_checkpoint`` for post-wrap shard restore, without model/Adam bytes.
        Legacy checkpoints return the shared payload plus
        ``'backend'`` = this rank's ``training_state_rank{r}.pt`` content, passed
        through untouched (``{}`` when absent and ``require_full_state`` is False).
        Accepts a checkpoint dir, its ``training_state.pt``, a ``file://`` URI,
        or an output/run root containing exactly one checkpoint lineage.
        """
        initialized = torch.distributed.is_initialized()
        original_path = path
        local_error = None
        state = None
        try:
            path = cls.resolve_resume_dir(path)
            if os.path.isfile(os.path.join(path, "training_state.json")):
                from specforge.training.nfc_checkpoint import read_manifest

                manifest = read_manifest(path)
                state = dict(
                    manifest["training"],
                    _nfc_checkpoint=path,
                    _nfc_common_sha256=manifest["common_sha256"],
                )
            else:
                state = torch.load(
                    os.path.join(path, STATE_FILE),
                    map_location=map_location,
                    weights_only=False,
                )
        except Exception as exc:
            local_error = exc

        world = torch.distributed.get_world_size() if initialized else 1
        if initialized and world > 1:
            descriptor = {
                "error": (
                    None
                    if local_error is None
                    else f"{type(local_error).__name__}: {local_error}"
                ),
                "checkpoint": (
                    None if local_error is not None else os.path.basename(path)
                ),
                "run_id": None if state is None else state.get("run_id"),
                "global_step": None if state is None else state.get("global_step"),
                "strategy": None if state is None else state.get("strategy"),
                "world_size": None if state is None else state.get("world_size"),
                "native_common_sha256": (
                    None if state is None else state.get("_nfc_common_sha256")
                ),
            }
            descriptors = [None] * world
            torch.distributed.all_gather_object(descriptors, descriptor)
            failures = [
                (rank, item["error"])
                for rank, item in enumerate(descriptors)
                if item["error"] is not None
            ]
            if failures:
                raise RuntimeError(
                    f"resume checkpoint {original_path!r} is not readable on "
                    "every rank: "
                    + "; ".join(f"rank {rank}: {error}" for rank, error in failures)
                )
            identities = {
                (
                    item["checkpoint"],
                    item["run_id"],
                    item["global_step"],
                    item["strategy"],
                    item["world_size"],
                    item["native_common_sha256"],
                )
                for item in descriptors
            }
            if len(identities) != 1:
                raise ValueError(
                    f"resume checkpoint {original_path!r} resolves to different "
                    f"training states across ranks: {descriptors}"
                )
        if local_error is not None:
            raise local_error
        assert state is not None

        rank = torch.distributed.get_rank() if initialized else 0
        saved_world = state.get("world_size")
        if "_nfc_checkpoint" in state:
            if require_full_state and saved_world != world:
                raise ValueError(
                    "native checkpoint restore requires the original world size"
                )
            state["backend"] = {}
            return state
        rank_path = os.path.join(path, cls._rank_file(rank))
        if os.path.exists(rank_path) and (saved_world is None or saved_world == world):
            state["backend"] = torch.load(
                rank_path, map_location=map_location, weights_only=False
            )
            if (
                state["backend"].get("optimizer") is None
                and "replicated_optimizer_state" in state
            ):
                state["backend"]["optimizer"] = state["replicated_optimizer_state"]
            return state
        state["backend"] = {}
        if require_full_state and int(state.get("global_step") or 0) > 0:
            raise ValueError(
                f"checkpoint {path} has no usable per-rank state for rank {rank} at "
                f"world_size={world} (written at world_size={saved_world}): resuming "
                f"training needs each rank's optimizer/RNG shard — resume at the "
                f"original world size, or pass require_full_state=False to restore "
                f"weights and counters only"
            )
        return state

    @classmethod
    def resolve_resume_dir(cls, path: str) -> str:
        """Resolve one unambiguous resume target to its checkpoint directory.

        A run output directory is convenient in declarative configs, but it
        must never choose between unrelated runs silently.  Prefer its single
        complete ``*-latest`` pointer; when symlinks are unavailable, fall back
        to the highest complete ``<run-id>-stepN`` directory only if all
        candidates belong to the same run id.
        """
        path = str(path)
        if path.startswith("file://"):
            path = path[len("file://") :]
        path = os.path.abspath(os.path.expanduser(path))
        if os.path.basename(path) in (STATE_FILE, "training_state.json"):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"checkpoint state file does not exist: {path}")
            path = os.path.dirname(path)
        if cls.is_complete(path):
            return path
        if not os.path.isdir(path):
            raise FileNotFoundError(f"checkpoint path does not exist: {path}")

        latest = []
        for candidate in glob.glob(os.path.join(glob.escape(path), "*-latest")):
            resolved = os.path.realpath(candidate)
            if cls.is_complete(resolved):
                latest.append(resolved)
        latest = sorted(set(latest))
        if len(latest) == 1:
            return latest[0]
        if len(latest) > 1:
            raise ValueError(
                f"resume root {path} contains multiple complete *-latest runs: "
                f"{latest}; select one checkpoint explicitly"
            )

        by_run: Dict[str, list[Tuple[int, str]]] = {}
        pattern = re.compile(r"^(.+)-step(\d+)$")
        for candidate in glob.glob(os.path.join(glob.escape(path), "*-step*")):
            match = pattern.match(os.path.basename(candidate))
            if match and cls.is_complete(candidate):
                by_run.setdefault(match.group(1), []).append(
                    (int(match.group(2)), candidate)
                )
        if not by_run:
            raise FileNotFoundError(f"no complete checkpoint found under {path}")
        if len(by_run) > 1:
            raise ValueError(
                f"resume root {path} contains multiple checkpoint runs: "
                f"{sorted(by_run)}; select one checkpoint explicitly"
            )
        candidates = next(iter(by_run.values()))
        return max(candidates, key=lambda item: item[0])[1]

    def latest_dir(self) -> Optional[str]:
        """Directory of the newest complete checkpoint, or None."""
        link = os.path.join(self.output_dir, f"{self.run_id}-latest")
        if os.path.islink(link):
            target = os.path.realpath(link)
            if self.is_complete(target):
                return target
        step = self._max_step_on_disk()
        return self.checkpoint_dir(step) if step is not None else None

    # -- internals ---------------------------------------------------------
    @staticmethod
    def is_rank0() -> bool:
        return (
            not torch.distributed.is_initialized()
        ) or torch.distributed.get_rank() == 0

    @staticmethod
    def _rank() -> int:
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    @staticmethod
    def _barrier() -> None:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    @staticmethod
    def _bcast_flag(flag: bool) -> bool:
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() == 1
        ):
            return bool(flag)
        box = [bool(flag)]
        torch.distributed.broadcast_object_list(box, src=0)
        return bool(box[0])

    @staticmethod
    def _atomic_save(obj: Any, path: str) -> None:
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    @staticmethod
    def _atomic_json(obj: Dict[str, Any], path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=2, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _point(self, name: str, ckpt_dir: str) -> None:
        link = os.path.join(self.output_dir, name)
        try:
            if os.path.islink(link):
                os.remove(link)
            elif os.path.isdir(link):
                logger.warning("removing non-symlink %s shadowing checkpoints", link)
                shutil.rmtree(link)
            elif os.path.exists(link):
                logger.warning("removing non-symlink %s shadowing checkpoints", link)
                os.remove(link)
            # Relative target survives relocating output_dir to another mount.
            os.symlink(os.path.basename(ckpt_dir), link)
        except OSError:
            return  # no symlink support: step dirs + best meta stay authoritative

    def _rotate(self, keep_step: int) -> None:
        from specforge.nsp.store import checkpoint_lock

        with checkpoint_lock(self.output_dir):
            self._rotate_unlocked(keep_step)

    def _rotate_unlocked(self, keep_step: int) -> None:
        if self.max_checkpoints <= 0:
            return
        dirs = sorted(self._all_checkpoints())
        for step, path in dirs[: -self.max_checkpoints]:
            if step == self.best_step or step == keep_step:
                continue
            try:
                shutil.rmtree(path)
            except OSError as exc:
                # Never raise between collectives; a leftover dir is harmless.
                logger.warning("rotation could not remove %s: %s", path, exc)

    def _step_dirs(self) -> Iterator[Tuple[int, str]]:
        # Raw {run_id}-step{N} dirs, complete or not (rewind must see both).
        pat = re.compile(rf"^{re.escape(self.run_id)}-step(\d+)$")
        pattern = os.path.join(
            glob.escape(self.output_dir), f"{glob.escape(self.run_id)}-step*"
        )
        for path in glob.glob(pattern):
            m = pat.match(os.path.basename(path))
            if m and os.path.isdir(path):
                yield int(m.group(1)), path

    def _all_checkpoints(self) -> Iterator[Tuple[int, str]]:
        # A truncated save without its completion marker is never a target.
        for step, path in self._step_dirs():
            if self.is_complete(path):
                yield step, path

    def _max_step_on_disk(self) -> Optional[int]:
        steps = [step for step, _ in self._all_checkpoints()]
        return max(steps) if steps else None


__all__ = ["CheckpointManager"]
