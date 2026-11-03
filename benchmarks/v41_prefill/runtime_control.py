"""Change experiment controls between fully drained benchmark phases."""

import importlib
import os
import re


def configure(runner, mode):
    match = re.fullmatch(
        r"codec([148])(?:-moe(8192|16384))?(-indexer)?(-deepinfer)?(-compact)?", mode
    )
    if mode != "baseline" and match is None:
        raise ValueError(f"Unsupported benchmark mode: {mode}")
    workers = match.group(1) if match else "1"
    moe_limit = int(match.group(2) or 0) if match else 0
    import bounded_moe

    bounded_moe.configure(moe_limit)
    import fused_indexer

    fused_indexer.configure(
        bool(match and (match.group(3) or match.group(4) or match.group(5))),
        deepinfer=bool(match and match.group(4)),
        compact=bool(match and match.group(5)),
    )
    os.environ["SPECFORGE_ZSTD_WORKERS"] = workers
    from specforge.data import feature_file, wire_codec

    importlib.reload(feature_file)
    importlib.reload(wire_codec)
    if os.environ.get("SGLANG_SPEC_CAPTURE_WIRE_ENCODING") == "bf16_fields_zstd_v2":
        from capture_validation import forbid_capture_hashes

        forbid_capture_hashes()
