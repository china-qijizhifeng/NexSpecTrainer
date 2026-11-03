"""Persist externally captured tensors after their Mooncake transfer.

One request is fetched to CPU, written atomically, then released before the
next request. The adapter restores tensors from the negotiated wire encoding
before writing. The offline reader decodes compressed records on demand during training.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import torch

from specforge.data.feature_file import (
    load_feature_file,
    require_hybrid_compression,
    require_zstandard,
    save_hybrid_features,
    save_zstd_features,
    validate_hybrid_options,
)
from specforge.inference.adapters.server_capture import ServerCaptureFailure
from specforge.inference.capture import verify_capture


def _atomic_json(path: Path, value) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
            temporary = f.name
            json.dump(value, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)


class ServerCaptureFileWriter:
    """A single-owner, synchronous server-to-offline-file capture client.

    ``capture`` and ``layout`` come from the selected algorithm's capture
    plan. ``adapter`` requests those offline field names directly from the
    server, while Mooncake continues carrying ordinary BF16 tensor bytes.
    """

    def __init__(
        self,
        output,
        *,
        capture,
        layout,
        algorithm: str,
        target_model_version: str,
        zstd: bool = False,
        zstd_level: int = 1,
        hybrid: bool = False,
        brotli_quality: int = 10,
        hybrid_chunk_size: int = 256 << 10,
        compression_workers: int = min(12, os.cpu_count() or 1),
        interleave_layers: bool = False,
        resume: bool = False,
        source_identity=None,
    ):
        if zstd and hybrid:
            raise ValueError("zstd and hybrid compression are mutually exclusive")
        if zstd:
            require_zstandard()
        if hybrid:
            require_hybrid_compression()
            validate_hybrid_options(
                brotli_quality,
                hybrid_chunk_size,
                compression_workers,
                layer_count=len(capture.aux_hidden_state_layer_ids),
            )
        if not 1 <= zstd_level <= 19:
            raise ValueError("zstd_level must be between 1 and 19")
        if (interleave_layers or hybrid) and (
            not layout.aux_feature or not capture.aux_hidden_state_layer_ids
        ):
            raise ValueError("layer compression requires an auxiliary feature")
        self.output = Path(output)
        self.capture, self.layout = capture, layout
        self.zstd, self.zstd_level = zstd, zstd_level
        self.hybrid = hybrid
        self.brotli_quality = brotli_quality
        self.hybrid_chunk_size = hybrid_chunk_size
        self.compression_workers = compression_workers
        self.interleave = (
            {layout.aux_feature: capture.aux_hidden_state_layer_ids}
            if interleave_layers or hybrid
            else {}
        )
        self.resume = resume
        self.suffix = ".ckpt.hybrid" if hybrid else (".ckpt.zst" if zstd else ".ckpt")
        self.plan = {
            "version": 1,
            "algorithm": algorithm,
            "source_identity": source_identity,
            "target_model_version": target_model_version,
            "features": sorted(capture.feature_names),
            "aux_layer_ids": list(capture.aux_hidden_state_layer_ids),
            "target_hidden_size": capture.target_hidden_size,
            "target_repr": capture.target_repr,
            "layout": {
                "aux_feature": layout.aux_feature,
                "last_hidden_feature": layout.last_hidden_feature,
                "passthrough": [list(p) for p in layout.passthrough],
            },
            "codec": (
                "conditional_hybrid"
                if hybrid
                else ("bf16_fields_zstd" if zstd else "torch")
            ),
            "zstd_level": zstd_level if zstd else None,
            "brotli_quality": brotli_quality if hybrid else None,
            "hybrid_chunk_size": hybrid_chunk_size if hybrid else None,
            "compression_workers": compression_workers if hybrid else None,
            "interleave_layers": interleave_layers or hybrid,
        }
        self._lock = None
        self._seen = set()

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=True)
        manifest = self.output / "capture-plan.json"
        if not self.resume and any(self.output.iterdir()):
            raise FileExistsError(f"capture output is not empty: {self.output}")
        lock = self.output / ".capture-writer.lock"
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        self._lock = lock
        os.close(fd)
        try:
            if manifest.exists():
                if json.loads(manifest.read_text()) != self.plan:
                    raise ValueError(
                        "resume capture plan does not match output manifest"
                    )
            elif self.resume and any(p != lock for p in self.output.iterdir()):
                raise ValueError(
                    "cannot resume a nonempty directory without capture-plan.json"
                )
            else:
                _atomic_json(manifest, self.plan)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_exc):
        if self._lock is not None:
            self._lock.unlink()
            self._lock = None

    def _validate(self, tensors, task):
        verify_capture(
            tensors,
            self.capture,
            sample_id=task.task_id,
            aux_feature_name=self.layout.aux_feature or "__no_aux__",
            target_feature_name=self.layout.last_hidden_feature or "__no_final__",
        )
        n = len(task.payload["input_ids"])
        record = {}
        for name, source in self.layout.passthrough:
            value = tensors[name]
            expected = torch.tensor(task.payload[source], dtype=torch.int64)
            if value.device.type != "cpu" or tuple(value.shape) != (1, n):
                raise ValueError(f"{name} must be CPU [1, {n}] before persistence")
            if value.dtype != torch.int64 or not torch.equal(value[0], expected):
                raise ValueError(f"{name} does not match requested {source}")
            record[name] = value[0]
        for name in (self.layout.aux_feature, self.layout.last_hidden_feature):
            if name is None:
                continue
            value = tensors[name]
            if (
                value.device.type != "cpu"
                or value.ndim != 3
                or tuple(value.shape[:2]) != (1, n)
            ):
                raise ValueError(f"{name} must be CPU [1, {n}, width]")
            if value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
                raise ValueError(f"{name} requires finite BF16 activations")
            record[name] = value
        return record

    def write(self, task, *, adapter, store):
        if self._lock is None:
            raise RuntimeError("use ServerCaptureFileWriter as a context manager")
        name = task.task_id
        if name == "capture-plan" or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name
        ):
            raise ValueError(
                "task id must be a simple filename of at most 128 characters"
            )
        if name in self._seen:
            raise ValueError(f"duplicate task id: {name}")
        self._seen.add(name)
        if task.target_model_version != self.plan["target_model_version"]:
            raise ValueError("task target model version does not match capture plan")
        identity = {
            "input_sha256": hashlib.sha256(
                json.dumps(task.payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "source_metadata": task.metadata,
        }
        final = self.output / (name + self.suffix)
        metadata = self.output / (name + ".json")
        if final.exists():
            if not self.resume:
                raise FileExistsError(final)
            if json.loads(metadata.read_text())["input"] != identity:
                raise ValueError(f"resume input does not match persisted sample {name}")
            # Decode/check all frames, shapes and tokens before trusting a skip.
            stored = load_feature_file(final)
            restored = {
                k: (
                    v.unsqueeze(0)
                    if k in {p[0] for p in self.layout.passthrough}
                    else v
                )
                for k, v in stored.items()
            }
            self._validate(restored, task)
            return {
                "sample": name,
                "status": "resumed",
                "tokens": len(task.payload["input_ids"]),
            }
        outcomes = adapter.produce_refs([task], capture=self.capture)
        if len(outcomes) != 1:
            raise RuntimeError("single-request capture must return exactly one outcome")
        ref = outcomes[0]
        if isinstance(ref, ServerCaptureFailure):
            raise RuntimeError(f"capture failed for {name}: {ref.reason}")
        handle = None
        temporary = None
        try:
            tensors, handle = store.get(ref, device="cpu")
            record = self._validate(tensors, task)
            fd, temporary = tempfile.mkstemp(dir=self.output, suffix=".partial")
            os.close(fd)
            if self.hybrid:
                save_hybrid_features(
                    record,
                    temporary,
                    quality=self.brotli_quality,
                    chunk_size=self.hybrid_chunk_size,
                    workers=self.compression_workers,
                    interleave_layers=self.interleave,
                )
            elif self.zstd:
                save_zstd_features(
                    record,
                    temporary,
                    level=self.zstd_level,
                    interleave_layers=self.interleave,
                )
            else:
                torch.save(record, temporary)
            with open(temporary, "rb") as f:
                os.fsync(f.fileno())
            result = {
                "sample": name,
                "status": "captured",
                "tokens": len(task.payload["input_ids"]),
                "tensor_bytes": sum(
                    v.numel() * v.element_size() for v in record.values()
                ),
                "file_bytes": os.path.getsize(temporary),
                **(
                    {
                        "wire_features": ref.metadata["wire_features"],
                        "wire_bytes": ref.estimated_bytes,
                        "wire_decoded_bytes": ref.metadata["decoded_bytes"],
                    }
                    if ref.metadata.get("wire_features")
                    else {}
                ),
            }
            _atomic_json(metadata, {"input": identity, "result": result})
            os.replace(temporary, final)
            temporary = None
            return result
        finally:
            if temporary is not None:
                os.unlink(temporary)
            if handle is not None:
                store.release(handle)
            else:
                store.abort(ref.sample_id, reason="capture-file-write-failed")
