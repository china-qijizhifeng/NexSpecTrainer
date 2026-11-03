"""Native NFC save/restore and a weights-only DCP reader for export."""

from __future__ import annotations

import hashlib
import json
import math
import os

import torch
import torch.distributed as dist

from specforge.training.nfc_state import FORMAT, MANIFEST, NativeState


def read_manifest(path):
    with open(os.path.join(path, MANIFEST), encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format") != FORMAT:
        raise ValueError(
            f"unsupported native checkpoint format: {manifest.get('format')}"
        )
    common_path = os.path.join(path, "nfc", "common.pt")
    with open(common_path, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != manifest["common_sha256"]:
        raise ValueError("native checkpoint common metadata checksum mismatch")
    # Core's common.pt contains only metadata: tensor payloads were extracted
    # into native descriptors. Read it directly before model construction so
    # tuple-valued provider/provenance contracts retain their exact types.
    common = torch.load(common_path, map_location="cpu", weights_only=True)
    training, layouts = common["training"], common["nfc_layout"]
    if len(layouts) != training["world_size"]:
        raise ValueError("native checkpoint manifest has an invalid rank count")
    if any(manifest[key] != training[key] for key in ("global_step", "run_id")):
        raise ValueError("native checkpoint completion identity mismatch")
    return {
        "format": FORMAT,
        "training": training,
        "layouts": layouts,
        "common_sha256": digest,
    }


def validate_layouts(layouts):
    """Reject missing/overlapping ownership and partially initialized Adam state."""
    first = layouts[0]
    for layout in layouts:
        if layout["model"].keys() != first["model"].keys():
            raise ValueError("draft checkpoint keys differ across ranks")
        if layout["optimizer_names"] != first["optimizer_names"]:
            raise ValueError("optimizer parameter order differs across ranks")
    for name, reference in first["model"].items():
        total = math.prod(reference["shape"])
        intervals = []
        for layout in layouts:
            item = layout["model"][name]
            if (item["shape"], item["dtype"]) != (
                reference["shape"],
                reference["dtype"],
            ):
                raise ValueError(f"global shape/dtype differs across ranks: {name}")
            if item["replica"] == 0 and item["numel"]:
                intervals.append((item["offset"], item["offset"] + item["numel"]))
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor or end > total:
                raise ValueError(f"incomplete or overlapping FSDP ownership: {name}")
            cursor = end
        if cursor != total:
            raise ValueError(f"incomplete FSDP ownership: {name}")
    for index, name in enumerate(first["optimizer_names"]):
        active = [
            layout["optimizer_slots"][index]
            for layout in layouts
            if layout["model"][name]["numel"]
        ]
        if active and any(fields != active[0] for fields in active):
            raise ValueError(
                f"Adam state is only initialized on some nonempty shards: {name}"
            )


class NativeCheckpoint:
    def __init__(self, manager, options):
        self.manager = manager
        from specforge.config.schema import NFCCheckpointConfig

        self.options = NFCCheckpointConfig.model_validate(options).model_dump()
        self.runtime = None

    def _checked(self, action):
        result, error = None, ""
        try:
            result = action()
        except Exception as exc:  # noqa: BLE001 -- errors must reach every rank
            error = f"{type(exc).__name__}: {exc}"
        self.manager._all_ok(error)
        return result

    def _runtime(self):
        if self.runtime is None:

            def create():
                from nexflashcheckpoint import MegatronCheckpointRuntime

                return MegatronCheckpointRuntime(**self.options)

            self.runtime = self._checked(create)
        return self.runtime

    def save(self, backend, state_filter, training, step):
        state = self._checked(lambda: NativeState(backend, state_filter))
        local = self._checked(state.layout)
        layouts = [None] * dist.get_world_size()
        dist.all_gather_object(layouts, {"layout": local, "training": training})

        def agree_training():
            if any(item["training"] != training for item in layouts):
                raise ValueError(
                    "native checkpoint training metadata differs across ranks"
                )

        self._checked(agree_training)
        layouts = [item["layout"] for item in layouts]
        self._checked(lambda: validate_layouts(layouts))
        native = self._checked(lambda: state.build(training, layouts))
        runtime = self._runtime()
        manager = self.manager
        path = manager.checkpoint_dir(step)

        def prepare():
            if manager.is_rank0():
                manager._rewind(step)
                os.makedirs(path, exist_ok=True)

        self._checked(prepare)
        ticket = runtime.save(
            native,
            os.path.join(path, "nfc"),
            training_mode="fullft",
            metadata={"format": FORMAT, "step": step},
        )
        # Both waits stay on the training lane. No parameter mutation is allowed
        # before capture completes; no latest/best publication before durability.
        ticket.wait("recoverable")

        def publish():
            if manager.is_rank0():
                with open(os.path.join(path, "nfc", "common.pt"), "rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                manager._atomic_json(
                    {
                        "format": FORMAT,
                        "common_sha256": digest,
                        "global_step": training["global_step"],
                        "run_id": training["run_id"],
                    },
                    os.path.join(path, MANIFEST),
                )
                manager._point(f"{manager.run_id}-latest", path)
                manager._rotate(keep_step=step)

        self._checked(publish)
        return path

    def restore(self, backend, state_filter, path):
        manifest = self._checked(lambda: read_manifest(path))
        digests = [None] * dist.get_world_size()
        dist.all_gather_object(digests, manifest["common_sha256"])

        def validate_identity():
            if len(set(digests)) != 1:
                raise ValueError("native checkpoint metadata differs across ranks")
            if manifest["training"]["world_size"] != dist.get_world_size():
                raise ValueError(
                    "native checkpoint restore requires the original world size"
                )

        self._checked(validate_identity)
        state = self._checked(lambda: NativeState(backend, state_filter))
        target = self._checked(
            lambda: state.build(
                manifest["training"], manifest["layouts"], restoring=True
            )
        )
        runtime = self._runtime()

        loaded = runtime.load(target, os.path.join(path, "nfc"))
        self._checked(lambda: state.apply(loaded, manifest["layouts"][dist.get_rank()]))

    def close(self):
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None


def load_draft_weights(path):
    """Read only draft tensors, on CPU, without initializing a process group.

    Export is allowed to materialize full draft weights. The training save and
    restore paths never call this reader or create an intermediate .pt snapshot.
    """
    from torch.distributed.checkpoint import FileSystemReader, load_state_dict
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    manifest = read_manifest(path)
    layout = manifest["layouts"][0]["model"]
    weights = {}
    for name, item in layout.items():
        dtype = getattr(torch, item["dtype"].removeprefix("torch."))
        weights[f"model/{name}"] = torch.empty(math.prod(item["shape"]), dtype=dtype)
    nonempty = {key: tensor for key, tensor in weights.items() if tensor.numel()}
    if nonempty:
        load_state_dict(
            nonempty,
            FileSystemReader(os.path.join(path, "nfc")),
            no_dist=True,
            planner=DefaultLoadPlanner(),
        )
    return {
        **manifest["training"],
        "draft_state_dict": {
            name: weights[f"model/{name}"].view(item["shape"])
            for name, item in layout.items()
        },
    }
