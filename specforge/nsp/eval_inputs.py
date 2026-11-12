"""Evaluation provenance and coverage checks before model computation."""

from pathlib import Path
import hashlib

from specforge.nsp.store import digest


def serving_checkpoint_identity(path):
    """External serving weights are attested by path plus metadata, not copied.

    This is explicitly not a content hash of every weight shard. Config/index
    contents are hashed; shard size/mtime inventories catch replacement during
    an evaluation without rereading hundreds of GB over the shared filesystem.
    """
    path = Path(path).resolve()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    if not files:
        raise ValueError("serving checkpoint is empty")
    rows = []
    for file in files:
        stat = file.stat()
        row = {"name": str(file.relative_to(path)) if path.is_dir() else file.name,
               "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if file.suffix == ".json":
            row["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
        rows.append(row)
    return {"format": "external-path-config-sha256-shard-stat-v1", "path": str(path),
            "files": rows, "fingerprint": digest(rows)}


def feature_inventory(path):
    from specforge.runtime.data_plane.offline_reader import list_feature_files

    files = []
    for name in list_feature_files(path):
        stat = Path(name).stat()
        files.append({"path": name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    if not files:
        raise ValueError("evaluation dataset has no feature files")
    return {
        "identity_format": "path-size-mtime-v1",
        "fingerprint": digest(files),
        "files": files,
    }


def checkpoint_metadata(path):
    import torch

    path = Path(path)
    native = path / "training_state.json" if path.is_dir() else path
    if native.name == "training_state.json" and native.is_file():
        from specforge.training.nfc_checkpoint import read_manifest

        state = read_manifest(str(native.parent))["training"]
    else:
        source = path / "training_state.pt" if path.is_dir() else path
        if not source.is_file() or source.name != "training_state.pt":
            return {}
        state = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    return {
        key: state[key]
        for key in (
            "global_step",
            "effective_total_steps",
            "specforge_model_provenance",
        )
        if key in state
    }


def validate_model_identity(metadata, bundle, output_dir):
    from specforge.algorithms.common.providers import MODEL_PROVENANCE_CONTRACT_KEY
    from specforge.training.provenance import model_provenance_for_resume_comparison

    saved = metadata.get(MODEL_PROVENANCE_CONTRACT_KEY)
    current = bundle.strategy_kwargs.resume_contract.get(MODEL_PROVENANCE_CONTRACT_KEY)
    if saved is None:
        return "unavailable-in-checkpoint"
    actual, expected = model_provenance_for_resume_comparison(
        current, saved, cache_root=str(output_dir)
    )
    if actual != expected:
        raise ValueError(
            "eval recipe target/draft/tokenizer contract does not match checkpoint provenance"
        )
    return "verified"
