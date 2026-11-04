"""Load the worker credential in memory, without putting it in the OS command line."""

import json
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    profile = json.loads((root / "candidate-profile.json").read_text())
    key = (root / "private" / "worker-key").read_text().strip()
    assert len(key) >= 40
    sys.argv = [
        "sglang.launch_server",
        "--model-path",
        profile["model_path"],
        "--served-model-name",
        profile["northgate_model"],
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        *profile["arguments"],
        "--api-key",
        key,
    ]
    runpy.run_module("sglang.launch_server", run_name="__main__")
