"""Temporary BP Dev composition until a forward+NFC runtime image is published."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

SERIALIZATION_SHA256 = (
    "5f745e742b63b4d4f4c406edda9c09605746bd52faa24c1b5fb4446676799a65"
)


def verify(forward, checkpoint, output):
    target = output / "megatron/core"
    files = {}
    for source, prefix in [
        (forward, ""),
        (checkpoint / "dist_checkpointing", "dist_checkpointing/"),
    ]:
        for path in sorted(source.rglob("*.py")):
            relative = path.relative_to(source)
            if "__pycache__" in relative.parts:
                continue
            if not prefix and relative.parts[0] == "dist_checkpointing":
                continue
            name = prefix + str(relative)
            destination = target / name
            if destination.read_bytes() != path.read_bytes():
                raise ValueError(f"Core source mismatch: {destination}")
            files[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    if set(files) != {str(p.relative_to(target)) for p in target.rglob("*.py")}:
        raise ValueError("Composed Core contains unexpected Python source files")
    if files.get("dist_checkpointing/serialization.py") != SERIALIZATION_SHA256:
        raise ValueError("Checkpoint Core differs from the validated Weaver image")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-core", type=Path, required=True)
    parser.add_argument("--checkpoint-core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    forward, checkpoint, output = (
        p.resolve() for p in (args.forward_core, args.checkpoint_core, args.output)
    )
    for source in (forward, checkpoint):
        if not (source / "__init__.py").is_file():
            raise ValueError(f"Not a Megatron Core source: {source}")
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Output must be separate from both shared sources")
    if not (checkpoint / "dist_checkpointing/serialization.py").is_file():
        raise ValueError("Checkpoint Core lacks serialization.py")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        temporary = Path(tempfile.mkdtemp(prefix=".training-core-", dir=output.parent))
        try:
            target = temporary / "megatron/core"
            shutil.copytree(
                forward,
                target,
                ignore=shutil.ignore_patterns(
                    "dist_checkpointing", "__pycache__", "*.pyc"
                ),
            )
            shutil.copytree(
                checkpoint / "dist_checkpointing",
                target / "dist_checkpointing",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            verify(forward, checkpoint, temporary)
            temporary.rename(output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    files = verify(forward, checkpoint, output)
    receipt = {
        "forward_source": str(forward),
        "checkpoint_source": str(checkpoint),
        "output": str(output),
        "files": files,
        "temporary_workaround": True,
    }
    (output / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(output), "verified_python_files": len(files)}))


if __name__ == "__main__":
    main()
