"""Run a command with persistent, explicitly namespaced GPU kernel caches.

Uses only the Python standard library; run with the same Python as the target
environment, before it imports SGLang, Triton, TileLang, or Torch. See
docs/sections/examples/kernel-cache.md for shared-storage and local-disk use.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def cache_environment(root: Path, namespace: str, inherited: dict[str, str]):
    """Return cache settings without importing GPU libraries or touching disk."""
    base = root.expanduser().resolve() / namespace
    directories = {
        "TRITON_CACHE_DIR": str(base / "triton"),
        "CUDA_CACHE_PATH": str(base / "cuda"),
        "XDG_CACHE_HOME": str(base / "xdg"),
        "TORCHINDUCTOR_CACHE_DIR": str(base / "torchinductor"),
        "TORCH_EXTENSIONS_DIR": str(base / "torch_extensions"),
        "TILELANG_CACHE_DIR": str(base / "tilelang"),
        # FlashInfer appends .cache/flashinfer/<version>/<architecture>.
        "FLASHINFER_WORKSPACE_BASE": str(base / "flashinfer"),
        "SGLANG_CACHE_DIR": str(base / "sglang"),
    }
    directories = {k: inherited.get(k, v) for k, v in directories.items()}
    sglang = Path(directories["SGLANG_CACHE_DIR"])
    directories.update(
        {
            "SGLANG_JIT_CACHE_DIR": inherited.get(
                "SGLANG_JIT_CACHE_DIR", str(sglang / "jit")
            ),
            "SGLANG_CUTE_AOT_CACHE_DIR": inherited.get(
                "SGLANG_CUTE_AOT_CACHE_DIR", str(sglang / "cute_aot")
            ),
        }
    )
    # SGLang overwrites DG_JIT_CACHE_DIR during import. Honor either spelling
    # supplied by the caller, and reject a conflict instead of losing an override.
    sg_dg = inherited.get("SGLANG_DG_CACHE_DIR")
    dg = inherited.get("DG_JIT_CACHE_DIR")
    if sg_dg is not None and dg is not None and sg_dg != dg:
        raise ValueError("SGLANG_DG_CACHE_DIR and DG_JIT_CACHE_DIR must agree")
    deepgemm = sg_dg if sg_dg is not None else dg
    if deepgemm is None:
        deepgemm = str(sglang / "deep_gemm")
    directories["SGLANG_DG_CACHE_DIR"] = deepgemm
    directories["DG_JIT_CACHE_DIR"] = deepgemm

    defaults = {
        "TRITON_CACHE_AUTOTUNING": "1",
        "TILELANG_DISABLE_CACHE": "0",
        "TILELANG_AUTO_TUNING_DISABLE_CACHE": "0",
        "SGLANG_FLASHINFER_AUTOTUNE_CACHE": "1",
        # With unchunked prefill, the full sweep visits M=1..65536 per N/K.
        # Fast warmup samples larger M values; missing shapes still JIT on use.
        "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "1",
    }
    settings = {
        **directories,
        **{k: inherited.get(k, v) for k, v in defaults.items()},
    }
    return settings, directories


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--namespace",
        required=True,
        help="Cache identity, e.g. sglang-<image-digest>-sm90; isolate training images",
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="Print only resolved cache settings as JSON; do not create directories",
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="-- command [args...]"
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.namespace):
        parser.error(
            "--namespace must be one path component starting with a letter/digit"
        )
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if args.print_env and command:
        parser.error("--print-env does not accept a command")
    if not args.print_env and not command:
        parser.error("a command is required after --")
    try:
        settings, directories = cache_environment(
            args.cache_root, args.namespace, dict(os.environ)
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.print_env:
        print(json.dumps(settings, indent=2, sort_keys=True))
        return 0
    try:
        for path in set(directories.values()):
            if path:  # An explicit empty CuTe AOT path disables that disk cache.
                Path(path).expanduser().mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"kernel cache: cannot prepare cache directory: {exc}", file=sys.stderr)
        return 1
    print(
        f"kernel cache: namespace={args.namespace} "
        f"SGLANG_CACHE_DIR={settings['SGLANG_CACHE_DIR']} "
        f"TILELANG_CACHE_DIR={settings['TILELANG_CACHE_DIR']} "
        f"TRITON_CACHE_DIR={settings['TRITON_CACHE_DIR']}",
        file=sys.stderr,
        flush=True,
    )
    # Replace the wrapper so signals, exit status, GPU arguments, and process
    # supervision belong to the real command. Never invoke a shell implicitly.
    try:
        os.execvpe(command[0], command, {**os.environ, **settings})
    except OSError as exc:
        print(f"kernel cache: cannot execute {command[0]!r}: {exc}", file=sys.stderr)
        return 127 if isinstance(exc, FileNotFoundError) else 126


if __name__ == "__main__":
    sys.exit(main())
