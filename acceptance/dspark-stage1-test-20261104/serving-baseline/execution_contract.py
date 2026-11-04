"""Verify the immutable local serving files without reading credentials or model payloads."""

import hashlib
import json
from pathlib import Path


def verify_execution(root):
    root = Path(root)
    manifest = json.loads((root / "execution-sha256.json").read_text())
    for name, digest in manifest.items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        path = root / relative
        assert path.is_file() and not path.is_symlink(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
