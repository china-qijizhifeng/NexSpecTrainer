"""Export trained native MTP weights as a separate SGLang model directory.

The target shards are symlinked; new MTP shards override only mtp.* index
entries. Run on a local, complete target snapshot. Source files are never edited.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from safetensors.torch import save_file

from .checkpoint import NativeCheckpoint, native_parts, quantize_like


def export_native(state, source, output):
    source, output = Path(source).resolve(), Path(output).absolute()
    if not source.is_dir():
        raise ValueError("export requires a complete local target snapshot")
    if output.exists():
        raise FileExistsError(output)
    checkpoint = NativeCheckpoint(source)
    mapping = {
        native: (key, sl)
        for key, tensor in state.items()
        for native, sl in native_parts(key, tensor.shape)
    }
    expected = {
        key
        for key in checkpoint.weight_map
        if key.startswith("mtp.") and not key.endswith(".scale")
    }
    if set(mapping) != expected:
        raise ValueError(
            f"native export key mismatch: missing={sorted(expected - set(mapping))}, extra={sorted(set(mapping) - expected)}"
        )
    missing = [
        name
        for name in set(checkpoint.weight_map.values())
        if not (source / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"target snapshot missing shards: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for item in source.iterdir():
            if item.name == "model.safetensors.index.json" or item.name.startswith("."):
                continue
            (temporary / item.name).symlink_to(item, target_is_directory=item.is_dir())
        weight_map = dict(checkpoint.weight_map)
        tensors, size, number = {}, 0, 0

        def flush():
            nonlocal tensors, size, number
            if not tensors:
                return
            filename = f"nexspec-mtp-{number:05d}.safetensors"
            if (temporary / filename).exists():
                raise ValueError(
                    "source already contains NexSpec MTP overlay; use original target"
                )
            save_file(tensors, temporary / filename, metadata={"format": "pt"})
            weight_map.update(dict.fromkeys(tensors, filename))
            tensors, size, number = {}, 0, number + 1

        for native, (key, sl) in mapping.items():
            value = state[key][sl].detach().cpu()
            raw = checkpoint.tensor(native)
            scale_key = native.removesuffix(".weight") + ".scale"
            scale = (
                checkpoint.tensor(scale_key)
                if native.endswith(".weight") and scale_key in checkpoint.weight_map
                else None
            )
            quantized, sf = quantize_like(value, raw, scale)
            if quantized.shape != raw.shape:
                raise ValueError(f"native export shape mismatch: {native}")
            tensors[native] = quantized.contiguous()
            size += quantized.numel() * quantized.element_size()
            if sf is not None:
                tensors[scale_key] = sf.contiguous()
                size += sf.numel() * sf.element_size()
            if size >= 512 * 1024**2:
                flush()
        flush()
        index = {**checkpoint.index, "weight_map": weight_map}
        (temporary / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n"
        )
        (temporary / "nexspec-export.json").write_text(
            json.dumps(
                {
                    "strategy": "deepseek_v41",
                    "target_snapshot": str(source),
                    "training_dtype": "bfloat16",
                    "export_quantization": "source template",
                    "requires_target_symlinks": True,
                },
                indent=2,
            )
            + "\n"
        )
        # Do not overwrite even an empty output directory created concurrently.
        output.mkdir()
        try:
            for item in temporary.iterdir():
                item.rename(output / item.name)
        except BaseException:
            shutil.rmtree(output)
            raise
    finally:
        shutil.rmtree(temporary)
    return output


def main():
    from specforge.export.checkpoint_io import resolve_training_state

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    state = resolve_training_state(args.checkpoint)
    export_native(state["draft_state_dict"], args.target, args.output)


if __name__ == "__main__":
    main()
