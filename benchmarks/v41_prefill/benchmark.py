"""Fixed-input, cache-isolated V4.1 prefill/capture benchmark.

All timings are seconds. Profiled forwards and unprofiled E2E are separate.
Actual trajectory tokens and model files are kept outside the repository.
"""

import argparse
import functools
import json
import os
import resource
import time
import uuid
from contextvars import ContextVar
from pathlib import Path


def main(args):
    os.environ["SPECFORGE_ZSTD_WORKERS"] = str(args.codec_workers)
    import requests
    import torch
    from capture_validation import forbid_capture_hashes, validate
    from offline_capture_layout import from_offline, to_offline

    import specforge.data.wire_codec as wire_codec
    from specforge.algorithms.common.hidden_states_data import (
        build_dspark_collator,
        build_dspark_offline_normalizer,
    )
    from specforge.data.feature_file import load_zstd_features, save_zstd_features
    from specforge.inference.adapters.server_capture import (
        ServerCaptureFailure,
        ServerCaptureSchema,
        SGLangServerCaptureAdapter,
    )
    from specforge.inference.capture import CaptureConfig
    from specforge.inference.capture_pipeline import capture_pipeline
    from specforge.runtime.contracts import PromptTask
    from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore

    torch.set_num_threads(args.cpu_threads)
    forbid_capture_hashes()
    revision = "1bae0e1088993b9b82de08f59de5be5a34e42170"
    rows = [json.loads(line) for line in args.samples.read_text().splitlines()]
    assert rows and all(
        len(r["input_ids"]) == args.tokens
        and r["metadata"]["model_revision"] == revision
        for r in rows
    )
    args.output.mkdir(parents=True, exist_ok=False)
    run_id = "prefill-" + uuid.uuid4().hex
    session = requests.Session()
    current_stage = ContextVar("capture_stage")

    def time_wrap(obj, name, label):
        old = getattr(obj, name)

        @functools.wraps(old)
        def call(*a, **k):
            begin = time.perf_counter()
            try:
                return old(*a, **k)
            finally:
                stage = current_stage.get()
                stage[label] = stage.get(label, 0.0) + time.perf_counter() - begin

        setattr(obj, name, call)

    time_wrap(wire_codec, "decode_wire_tensor", "decode_s")
    time_wrap(MooncakeFeatureStore, "_fetch_tensor", "get_wire_s")

    def post(url, json_body, timeout):
        begin = time.perf_counter()
        response = session.post(url, json=json_body, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        stage = current_stage.get()
        stage["http_request_s"] = time.perf_counter() - begin
        stage["server_meta"] = [
            {
                k: v
                for k, v in x.get("meta_info", {}).items()
                if k
                in (
                    "cached_tokens",
                    "prompt_tokens",
                    "completion_tokens",
                    "e2e_latency",
                    "time_stats",
                    "id",
                )
            }
            for x in (data if isinstance(data, list) else [data])
        ]
        return data

    capture = CaptureConfig.from_strategy(
        ["input_ids", "loss_mask", "hidden_states", "target_last_hidden_states"],
        [37, 38, 39],
        target_repr="hidden_state",
        target_hidden_size=5120,
        target_vocab_size=129280,
        draft_vocab_size=129280,
    )
    # The source retains two results at depth two. Use the codec's worst-case
    # wire bound, not the measured compression ratio, before issuing requests.
    feature_bytes = [args.tokens * width * 2 for width in (15360, 5120)]
    feature_bytes += [args.tokens * 8] * 2
    wire_bound = sum(n + n // 8 + (1 << 20) for n in feature_bytes)
    if args.pipeline_depth * wire_bound > 8 << 30:
        raise ValueError("Pipeline exceeds the dedicated 8 GiB source capacity")
    store = MooncakeFeatureStore(
        store_id=run_id,
        setup_kwargs={
            "local_hostname": os.environ["MOONCAKE_LOCAL_HOSTNAME"],
            "metadata_server": "http://127.0.0.1:35880/metadata",
            "master_server_addr": "127.0.0.1:35551",
            "protocol": "rdma",
            "rdma_devices": "mlx5_00",
            "global_segment_size": 0,
            "local_buffer_size": 8 << 30,
        },
    )
    adapter = SGLangServerCaptureAdapter(
        args.server,
        store,
        run_id=run_id,
        algorithm="deepseek_v41",
        target_model_version=revision,
        server_capture="auto",
        timeout_s=1800,
        wire_encoding="bf16_fields_zstd_v2",
        wire_max_object_bytes=8 << 30,
        wire_max_total_bytes=8 << 30,
        post_fn=post,
        schema=ServerCaptureSchema(
            aux_feature="hidden_states",
            last_hidden_feature="target_last_hidden_states",
            passthrough=(
                ("input_ids", "input_ids", ()),
                ("loss_mask", "loss_mask", ()),
            ),
        ),
    )
    root = Path(os.environ["PREFILL_ROOT"])

    def control(tag, profile, detail=False):
        p = root / "control.tmp"
        p.write_text(
            json.dumps(
                {
                    "tag": args.output.name + "/" + tag,
                    "profile": profile,
                    "runtime_mode": "codec"
                    + str(args.codec_workers)
                    + (f"-moe{args.moe_chunk_tokens}" if args.moe_chunk_tokens else "")
                    + ("-indexer" if args.fused_indexer else "")
                    + ("-deepinfer" if args.deepinfer_indexer else "")
                    + ("-compact" if args.compact_candidates else ""),
                    "detail": detail,
                }
            )
        )
        p.replace(root / "control.json")

    def resources():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        cgroup = Path("/sys/fs/cgroup")
        return {
            "client_cpu_s": usage.ru_utime + usage.ru_stime,
            "client_maxrss_kib": usage.ru_maxrss,
            "cgroup": {
                name: (cgroup / name).read_text()
                for name in ("memory.current", "memory.peak", "cpu.stat")
                if (cgroup / name).exists()
            },
        }

    resources_before = resources()
    records = []
    phase_wall = {}

    def produce(item):
        row, phase, index, publish, write = item
        stage = {}
        current_stage.set(stage)
        started_unix_s = time.time()
        begin = time.perf_counter()
        name = f"{phase}-{index:03d}"
        if not publish:
            post(
                args.server + "/generate",
                {
                    "input_ids": row["input_ids"],
                    "extra_key": uuid.uuid4().hex,
                    "sampling_params": {"temperature": 0.0, "max_new_tokens": 0},
                },
                1800,
            )
        else:
            task = PromptTask(
                task_id=name,
                run_id=run_id,
                source_id=row["id"],
                payload={"input_ids": row["input_ids"], "loss_mask": row["loss_mask"]},
                max_length=args.tokens,
                target_model_version=revision,
                metadata={"num_tokens": args.tokens, "tokenizer_version": revision},
            )
            ref = adapter.produce_refs([task], capture=capture)[0]
            if isinstance(ref, ServerCaptureFailure):
                raise RuntimeError(ref.reason)
        return {
            "ref": ref if publish else None,
            "stage": stage,
            "started_unix_s": started_unix_s,
            "begin": begin,
        }

    def discard(item, produced):
        ref = produced["ref"]
        if ref is not None:
            store.abort(ref.sample_id, reason="pipeline-stopped")
            store.drain_sample_removals([ref.sample_id])

    def consume(item, produced):
        row, phase, index, publish, write = item
        stage = produced["stage"]
        current_stage.set(stage)
        started_unix_s, begin = produced["started_unix_s"], produced["begin"]
        name = f"{phase}-{index:03d}"
        ref = produced["ref"]
        if ref is not None:
            handle = None
            try:
                t = time.perf_counter()
                tensors, handle = store.get(ref, device="cpu")
                stage["get_decode_s"] = time.perf_counter() - t
                t = time.perf_counter()
                features = validate(tensors, row)
                stage["validate_s"] = time.perf_counter() - t
                stage["features"] = features
                stage["wire_bytes"] = ref.estimated_bytes
                if write:
                    t = time.perf_counter()
                    path = args.output / (name + ".ckpt.zst")
                    save_zstd_features(
                        to_offline(tensors), path, level=1, checksums=False
                    )
                    with path.open("rb") as f:
                        os.fsync(f.fileno())
                    stage["encode_write_fsync_s"] = time.perf_counter() - t
                    stage["disk_bytes"] = path.stat().st_size
            finally:
                t = time.perf_counter()
                if handle is not None:
                    store.release(handle)
                else:
                    store.abort(ref.sample_id)
                store.drain_sample_removals([ref.sample_id])
                stage["release_s"] = time.perf_counter() - t
        item = {
            "phase": phase,
            "started_unix_s": started_unix_s,
            "index": index,
            "source_id": row["id"],
            "tokens": args.tokens,
            "total_s": time.perf_counter() - begin,
            **stage,
        }
        records.append(item)
        with (args.output / "requests.jsonl").open("a") as f:
            f.write(json.dumps(item) + "\n")
        print(
            json.dumps({k: item[k] for k in ("phase", "index", "total_s")}), flush=True
        )

    try:
        # Resolve capability and establish RDMA before the measured sections.
        control("warmup", False)
        for i in range(args.warmup):
            item = (rows[i % len(rows)], "warmup", i, True, False)
            consume(item, produce(item))
        if not args.throughput_only:
            for phase, publish in [
                ("profile-no-publication", False),
                ("profile-capture", True),
            ]:
                control(phase, True, True)
                for i in range(args.profile_requests):
                    item = (rows[i % len(rows)], phase, i, publish, False)
                    consume(item, produce(item))
        for phase, publish, write in [
            ("no-publication", False, False),
            ("capture-no-disk", True, False),
            ("capture-disk", True, True),
        ]:
            control(phase, False)
            phase_begin = time.perf_counter()
            items = (
                (rows[i % len(rows)], phase, i, publish, write)
                for i in range(args.repeats * len(rows))
            )
            for _ in capture_pipeline(
                items,
                produce=produce,
                consume=consume,
                discard=discard,
                depth=args.pipeline_depth if publish else 1,
            ):
                pass
            phase_wall[phase] = time.perf_counter() - phase_begin
        # Read-back is outside throughput timing; validates the training file reader.
        checked = 0
        for item in records:
            path = args.output / (f"{item['phase']}-{item['index']:03d}.ckpt.zst")
            if path.exists():
                raw = load_zstd_features(path)
                normalized = build_dspark_collator()(
                    [build_dspark_offline_normalizer(args.tokens)(raw)]
                )
                assert normalized["hidden_states"].shape == (1, args.tokens, 15360)
                assert normalized["target_last_hidden_states"].shape == (
                    1,
                    args.tokens,
                    5120,
                )
                tensors = from_offline(raw)
                observed = validate(tensors, rows[item["index"] % len(rows)])
                assert observed == item["features"], path
                checked += 1
        summary = {}
        for phase in sorted({r["phase"] for r in records}):
            data = [r for r in records if r["phase"] == phase]
            times = sorted(r["total_s"] for r in data)
            summary[phase] = {
                "requests": len(data),
                "tokens": args.tokens * len(data),
                "total_s": phase_wall.get(phase, sum(times)),
                "tokens_per_s": args.tokens
                * len(data)
                / phase_wall.get(phase, sum(times)),
                "p50_s": float(__import__("statistics").median(times)),
                "p95_s": times[
                    min(len(times) - 1, __import__("math").ceil(0.95 * len(times)) - 1)
                ],
                "mean_stage_s": {
                    k: sum(r.get(k, 0.0) for r in data) / len(data)
                    for k in (
                        "http_request_s",
                        "get_wire_s",
                        "decode_s",
                        "validate_s",
                        "encode_write_fsync_s",
                        "release_s",
                    )
                },
            }
        receipt = {
            "model_revision": revision,
            "tokens_per_request": args.tokens,
            "run_id": run_id,
            "resources_before": resources_before,
            "resources_after": resources(),
            "store_health": store.health(),
            "samples_path": str(args.samples.resolve()),
            "hash_free": True,
            "pipeline_depth": args.pipeline_depth,
            "wire_encoding": "bf16_fields_zstd_v2",
            "cpu_threads": args.cpu_threads,
            "codec_workers": args.codec_workers,
            "moe_chunk_tokens": args.moe_chunk_tokens,
            "fused_indexer": args.fused_indexer,
            "deepinfer_indexer": args.deepinfer_indexer,
            "compact_candidates": args.compact_candidates,
            "checked_files": checked,
            "summary": summary,
            "note": "no-publication retains global aux capture allocation; no Mooncake publication. Profiled timing synchronizes CUDA; throughput phases do not.",
        }
        (args.output / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        control("idle", False)
        try:
            store.discard_external_attempts(reason="capture-benchmark-stopped")
            store.drain_pending_removals()
        finally:
            store.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", default="http://127.0.0.1:30000")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, choices=(4096, 16384, 65536), default=4096)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--profile-requests", type=int, default=4)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--throughput-only", action="store_true")
    p.add_argument("--pipeline-depth", type=int, choices=(1, 2), default=1)
    p.add_argument("--moe-chunk-tokens", type=int, choices=(0, 8192, 16384), default=0)
    p.add_argument("--fused-indexer", action="store_true")
    p.add_argument(
        "--deepinfer-indexer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the validated Hopper DeepInfer scorer (default: enabled)",
    )
    p.add_argument(
        "--compact-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store only candidate scores before top-k (default: enabled)",
    )
    p.add_argument("--codec-workers", type=int, choices=(1, 4, 8), default=1)
    main(p.parse_args())
