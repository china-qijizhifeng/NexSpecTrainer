"""Inspect the pinned serving image on CPU before choosing its deployment entrypoints."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "image-probe"
OUT.mkdir(exist_ok=True)


def main():
    assert os.environ["NODE_NAME"] == "node040"
    assert os.getuid() == 1006
    versions = {}
    for name in ("sglang", "sglang-router", "torch", "protobuf", "flashinfer-python"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    assert "38d9e4a3f" in versions["sglang"], versions
    spec = importlib.util.find_spec("sglang")
    package = Path(spec.origin).parent
    names = [
        "srt/server_args.py",
        "srt/entrypoints/http_server.py",
        "srt/utils/auth.py",
        "srt/arg_groups/serving.py",
    ]
    sources = {}
    for name in names:
        path = package / name
        if path.is_file():
            data = path.read_bytes()
            sources[name] = hashlib.sha256(data).hexdigest()
    router = importlib.util.find_spec("sglang_router")
    router_files = []
    if router is not None:
        folder = Path(router.origin).parent
        destination = OUT / "router-source"
        destination.mkdir(exist_ok=True)
        for file in folder.rglob("*.py"):
            rel = file.relative_to(folder)
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(file.read_bytes())
            router_files.append(str(rel))
        completed = subprocess.run(
            [sys.executable, "-m", "sglang_router.launch_router", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
        (OUT / "router-help.txt").write_text(completed.stdout)
        assert completed.returncode == 0, "installed router help failed"
    assert not any("SIMULATE" in key for key in os.environ)
    result = {
        "passed": True,
        "scope": "CPU image/entrypoint inspection only; no model forward or serving acceptance",
        "time": time.time(),
        "node": os.environ["NODE_NAME"],
        "uid": os.getuid(),
        "python": sys.version,
        "versions": versions,
        "source_sha256": sources,
        "router_installed": router is not None,
        "router_source_files": router_files,
        "gpu_requested": 0,
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
