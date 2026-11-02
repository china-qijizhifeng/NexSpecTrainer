"""Capture a running SGLang server through Mooncake into offline feature files.

The client loads model configuration only. It never executes target inference.
Input JSONL contains stable ``id``, ``input_ids``, ``loss_mask`` and optional
``metadata``. Tokens are already templated; overlength records fail rather than
being silently truncated. See docs/sections/examples/server-capture-to-disk.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from pathlib import Path

from specforge.runtime.contracts import PromptTask


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Offline training YAML (model, draft, strategy, max_length)",
    )
    parser.add_argument(
        "--server", required=True, help="Existing SGLang capture server URL"
    )
    parser.add_argument(
        "--samples",
        type=Path,
        required=True,
        help="Pre-tokenized JSONL; one complete episode per record",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-model-version",
        required=True,
        help="Pinned identity of the actual server checkpoint",
    )
    parser.add_argument(
        "--server-capture", choices=("auto", "plugin", "patch"), default="auto"
    )
    parser.add_argument("--timeout-s", type=float, default=3600)
    parser.add_argument("--client-buffer-bytes", type=int, default=4 << 30)
    parser.add_argument(
        "--server-buffer-bytes",
        type=int,
        default=4 << 30,
        help="Declared source buffer capacity; configure the SGLang server to match",
    )
    parser.add_argument(
        "--zstd",
        action="store_true",
        help="Client-side lossless BF16 fields + Zstd persistence",
    )
    parser.add_argument("--zstd-level", type=int, choices=range(1, 20), default=1)
    parser.add_argument("--interleave-layers", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.interleave_layers and not args.zstd:
        parser.error("--interleave-layers requires --zstd")
    if min(args.client_buffer_bytes, args.server_buffer_bytes) <= 0:
        parser.error("buffer capacities must be positive")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s must be finite and positive")
    if args.zstd:
        from specforge.data.feature_file import require_zstandard

        require_zstandard()  # before model config loading or Mooncake allocation
    return args


def iter_tasks(path, *, cfg, run_id, target_version, loss_mask_filter):
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"{path}:{line_number}: stable string id is required")
            ids, mask = row.get("input_ids"), row.get("loss_mask")
            if (
                not isinstance(ids, list)
                or not isinstance(mask, list)
                or not ids
                or len(ids) != len(mask)
            ):
                raise ValueError(
                    f"{row['id']}: nonempty equally sized input_ids/loss_mask required"
                )
            if any(type(t) is not int or t < 0 for t in ids) or any(
                type(t) is not int or t not in (0, 1) for t in mask
            ):
                raise ValueError(
                    f"{row['id']}: token ids must be nonnegative integers and mask binary integers"
                )
            if len(ids) > cfg.data.max_length:
                raise ValueError(
                    f"{row['id']}: {len(ids)} tokens exceeds max_length={cfg.data.max_length}; refusing truncation"
                )
            if not any(mask) or (
                loss_mask_filter is not None and not loss_mask_filter(mask)
            ):
                raise ValueError(
                    f"{row['id']}: sample does not satisfy strategy supervision requirements"
                )
            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"{row['id']}: metadata must be an object")
            if metadata.get("model_revision", target_version) != target_version:
                raise ValueError(
                    f"{row['id']}: model_revision differs from --target-model-version"
                )
            yield PromptTask(
                task_id=row["id"],
                run_id=run_id,
                source_id=row["id"],
                payload={"input_ids": ids, "loss_mask": mask},
                max_length=len(ids),
                target_model_version=target_version,
                metadata=metadata,
            )


def check_capacity(tokens, *, capture, layout, client_bytes, server_bytes):
    """Conservative single-record BF16 and INT64 transfer budget, no compression credit."""
    widths = (
        [capture.target_hidden_size * len(capture.aux_hidden_state_layer_ids)]
        if layout.aux_feature
        else []
    )
    if layout.last_hidden_feature:
        widths.append(capture.target_hidden_size)
    feature_bytes = [tokens * width * 2 for width in widths]
    feature_bytes.extend(tokens * 8 for _ in layout.passthrough)
    total = sum(feature_bytes)
    if total > min(client_bytes, server_bytes):
        raise ValueError(
            f"uncompressed capture needs {total} bytes; client/server declared buffers "
            f"are {client_bytes}/{server_bytes}. Enlarge the client buffer and the "
            "server MOONCAKE_GLOBAL_SEGMENT_SIZE / MOONCAKE_LOCAL_BUFFER_SIZE; "
            "disk compression does not reduce this transport allocation."
        )
    if total > 8 << 30:
        raise ValueError("capture exceeds the codec's 8 GiB decoded-record limit")
    return total


def main(argv=None):
    args = parse_args(argv)
    from specforge.application import resolve_offline_capture
    from specforge.config import load_config
    from specforge.inference.adapters.server_capture import (
        ServerCaptureSchema,
        SGLangServerCaptureAdapter,
    )
    from specforge.inference.capture import CaptureConfig
    from specforge.inference.server_capture_writer import ServerCaptureFileWriter
    from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore

    cfg = load_config(args.config)
    if cfg.model.input_modality != "text":
        raise ValueError(
            "server-to-file preparation currently supports pre-tokenized text only"
        )
    plan = resolve_offline_capture(cfg)
    layout = plan.layout
    from specforge.training.capture_contract import resolve_server_capture_contract

    contract = resolve_server_capture_contract(cfg, algorithm=plan.run.algorithm)
    capture = CaptureConfig.from_strategy(
        layout.output_names,
        contract.aux_layer_ids,
        target_repr="hidden_state",
        target_hidden_size=contract.target_hidden_size,
        target_vocab_size=contract.target_vocab_size,
        draft_vocab_size=contract.draft_vocab_size,
    )
    with args.samples.open("rb") as source:
        source_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    run_id = "capture-files-" + uuid.uuid4().hex
    with ServerCaptureFileWriter(
        args.output,
        capture=capture,
        layout=layout,
        algorithm=plan.run.algorithm.name,
        target_model_version=args.target_model_version,
        zstd=args.zstd,
        zstd_level=args.zstd_level,
        interleave_layers=args.interleave_layers,
        resume=args.resume,
        source_identity={
            "samples_sha256": source_sha256,
        },
    ) as writer:
        store = MooncakeFeatureStore(
            store_id=run_id,
            setup_kwargs={
                "local_hostname": os.environ.get(
                    "MOONCAKE_LOCAL_HOSTNAME", "127.0.0.1"
                ),
                "metadata_server": os.environ["MOONCAKE_METADATA_SERVER"],
                "master_server_addr": os.environ["MOONCAKE_MASTER_SERVER_ADDR"],
                "protocol": os.environ.get("MOONCAKE_PROTOCOL", "tcp"),
                "rdma_devices": os.environ.get("MOONCAKE_RDMA_DEVICES", ""),
                "global_segment_size": 0,
                "local_buffer_size": args.client_buffer_bytes,
            },
            receive_buffers="pageable",
        )
        try:
            adapter = SGLangServerCaptureAdapter(
                args.server,
                store,
                run_id=run_id,
                algorithm=plan.run.algorithm.name,
                target_model_version=args.target_model_version,
                timeout_s=args.timeout_s,
                server_capture=args.server_capture,
                schema=ServerCaptureSchema(
                    aux_feature=layout.aux_feature,
                    last_hidden_feature=layout.last_hidden_feature,
                    passthrough=tuple(
                        (name, source, ()) for name, source in layout.passthrough
                    ),
                ),
            )
            count = 0
            for task in iter_tasks(
                args.samples,
                cfg=cfg,
                run_id=run_id,
                target_version=args.target_model_version,
                loss_mask_filter=plan.loss_mask_filter,
            ):
                check_capacity(
                    len(task.payload["input_ids"]),
                    capture=capture,
                    layout=layout,
                    client_bytes=args.client_buffer_bytes,
                    server_bytes=args.server_buffer_bytes,
                )
                print(
                    json.dumps(writer.write(task, adapter=adapter, store=store)),
                    flush=True,
                )
                count += 1
            if not count:
                raise ValueError("sample file contains no records")
        finally:
            store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
