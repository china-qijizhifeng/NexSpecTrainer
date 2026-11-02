"""Add hashed wheels/sources while leaving Weaver's Torch and TE in place."""

import hashlib
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

BUILD = Path("/build")
ROOT = Path("/opt/nexspec")
ROOT.mkdir(exist_ok=True)
(ROOT / "evidence").mkdir(exist_ok=True)


def verify(path, digest):
    with path.open("rb") as file:
        assert hashlib.file_digest(file, "sha256").hexdigest() == digest, path


sources = json.loads((BUILD / "source-hashes.json").read_text())
destinations = {
    "trainer-current.tar.gz": "src",
    "tile-kernels.tar.gz": "tile",
    "megatron-csa.tar.gz": "megatron-src",
}
assert sources.keys() == destinations.keys()
for name, digest in sources.items():
    path = BUILD / name
    verify(path, digest)
    destination = ROOT / destinations[name]
    destination.mkdir(exist_ok=True)
    with tarfile.open(path) as archive:
        archive.extractall(destination, filter="data")

for manifest, folder in [
    ("overlay-packages.json", "wheelhouse"),
    ("native-wheels.json", "native-wheels"),
]:
    for row in json.loads((BUILD / manifest).read_text()):
        path = BUILD / folder / row["filename"]
        verify(path, row["sha256"])
        with zipfile.ZipFile(path) as archive:
            archive.extractall(ROOT / "overlay")
    (ROOT / "evidence" / manifest).write_text((BUILD / manifest).read_text())
(ROOT / "evidence/source-hashes.json").write_text(
    (BUILD / "source-hashes.json").read_text()
)
(ROOT / "bin").mkdir(exist_ok=True)
cli = ROOT / "bin/specforge"
cli.write_text('#!/bin/sh\nexec python -m specforge.cli "$@"\n')
cli.chmod(0o755)

# A fresh interpreter sees the final PYTHONPATH and confirms binary provenance.
subprocess.run(
    [
        "python",
        "-c",
        (
            "import torch, transformer_engine.pytorch, transformer_engine_torch as te; "
            "import importlib.metadata as m; "
            "assert torch.__version__ == '2.10.0a0+b558c986e8.nv25.11'; "
            "assert torch.__file__.startswith('/usr/local/lib/python3.12/dist-packages/torch/'); "
            "assert m.version('transformer-engine') == '2.20.2+6ea2a74'; "
            "assert te.__file__.startswith('/opt/venv/lib/python3.12/site-packages/transformer_engine/'); "
            "print('Preserved Weaver Torch/TE binaries')"
        ),
    ],
    check=True,
)
