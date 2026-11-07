"""Evaluation provenance and coverage checks before model computation."""

from pathlib import Path

from specforge.nsp.store import digest


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
