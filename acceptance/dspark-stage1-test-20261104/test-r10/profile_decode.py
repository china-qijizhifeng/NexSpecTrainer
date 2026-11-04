"""Bounded validation-only profile; traces stay private, publish aggregate kernels."""

import collections
import gzip
import json
import os

from common import sha, write


def summarize_trace(path):
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    kernels = collections.defaultdict(lambda: {"count": 0, "total_us": 0.0})
    runtime = collections.defaultdict(lambda: {"count": 0, "total_us": 0.0})
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat", "")
        dest = (
            kernels
            if category == "kernel"
            else runtime if category == "cuda_runtime" else None
        )
        if dest is None:
            continue
        duration = float(event.get("dur", 0))
        if duration < 0:
            continue
        row = dest[event["name"]]
        row["count"] += 1
        row["total_us"] += duration
    top = lambda rows: [
        {"name": name, **value}
        for name, value in sorted(rows.items(), key=lambda pair: -pair[1]["total_us"])[
            :20
        ]
    ]
    return {
        "trace_sha256": sha(path),
        "bytes": path.stat().st_size,
        "kernel_time_sum_us": sum(row["total_us"] for row in kernels.values()),
        "kernel_calls": sum(row["count"] for row in kernels.values()),
        "top_kernels": top(kernels),
        "top_cuda_runtime": top(runtime),
        "scope": "Duration sums overlap across streams; not wall time or causal attribution",
    }


def profile_validation(folder, row, generate, request):
    traces = folder / "private-profile"
    traces.mkdir(mode=0o700)
    os.chmod(traces, 0o700)
    started = False

    def arm():
        nonlocal started
        response = request(
            "/start_profile",
            {
                "output_dir": str(traces),
                "num_steps": 8,
                "activities": ["CPU", "GPU"],
                "with_stack": False,
                "record_shapes": False,
                "merge_profiles": False,
                "profile_id": "r10-validation-decode",
            },
            timeout=120,
            raw=True,
        )
        assert response.startswith("Start profiling."), "bounded profiler did not start"
        started = True

    try:
        result = generate(
            row, 64, folder / "profile-validation.json", on_first_token=arm
        )
    finally:
        # An automatically stopped profile rejects a second stop with HTTP500.
        # Generation may also EOS before8 steps, when an explicit stop is required.
        if started and len(list(traces.glob("*.trace.json.gz"))) < 8:
            response = request("/stop_profile", {}, timeout=180, raw=True)
            assert response.startswith("Stop profiling.")
    assert started, "validation output had no token event"
    paths = sorted(traces.glob("*.trace.json.gz"))
    assert len(paths) == 8, "expected one bounded trace per TP rank"
    summaries = []
    for path in paths:
        os.chmod(path, 0o600)
        summaries.append(summarize_trace(path))
    write(
        folder / "decode-profile-summary.json",
        {
            "passed": True,
            "validation_only": True,
            "excluded_from_benchmark_metrics": True,
            "requested_forward_steps": 8,
            "request_input_sha256": result["input_sha256"],
            "output_tokens": len(result["output_ids"]),
            "trace_count": len(paths),
            "traces": summaries,
        },
    )
