"""Summarize actual forward measurements and fixed-input capture comparisons."""

import argparse
import json
import statistics
from pathlib import Path


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize(root, experiments):
    forwards = {}
    for path in sorted(root.glob("forward-rank*.jsonl")):
        for row in lines(path):
            forwards.setdefault(row["tag"], {}).setdefault(row["sequence"], []).append(
                row
            )
    forward_summary = {}
    for tag, groups in forwards.items():
        assert all(
            len(batch) == 8 and len({r["rank"] for r in batch}) == 8
            for batch in groups.values()
        ), tag
        lengths = {r["tokens"] for batch in groups.values() for r in batch}
        assert len(lengths) == 1, (tag, lengths)
        tokens = lengths.pop()
        times = [max(r["forward_cuda_ms"] for r in batch) for batch in groups.values()]
        rank0 = [next(r for r in batch if r["rank"] == 0) for batch in groups.values()]
        names = sorted({name for r in rank0 for name in r["groups"]})
        forward_summary[tag] = {
            "samples": len(times),
            "tokens_per_request": tokens,
            "max_rank_cuda_ms_p50": statistics.median(times),
            "max_rank_cuda_ms_mean": statistics.mean(times),
            "tokens_per_forward_second": tokens * len(times) / (sum(times) / 1000),
            "max_rank_wall_ms_p50": statistics.median(
                max(r["forward_wall_ms"] for r in b) for b in groups.values()
            ),
            "rank0_groups_mean_ms": {
                name: statistics.mean(
                    r["groups"].get(name, {}).get("cuda_ms", 0) for r in rank0
                )
                for name in names
            },
            "gpu_peak_allocated_bytes_max": max(
                r["gpu_peak_allocated"] for b in groups.values() for r in b
            ),
        }
    captures = {}
    checks = {}
    baselines = {}
    for name in experiments:
        folder = root / name
        captures[name] = json.loads((folder / "summary.json").read_text())
        data = [
            r for r in lines(folder / "requests.jsonl") if r["phase"] == "capture-disk"
        ]
        by_id = {}
        for r in data:
            by_id.setdefault(r["source_id"], []).append(r["features"])
        checks[name] = {
            "samples": len(by_id),
            "repeat_bitwise_equal": all(
                all(x == rs[0] for x in rs) for rs in by_id.values()
            ),
        }
        if captures[name].get("hash_free"):
            # Shape/dtype metadata is not evidence of content equality.
            checks[name] = {
                "samples": len(by_id),
                "tokens_per_request": captures[name]["tokens_per_request"],
                "content_equality": "requires direct tensor comparison receipt",
            }
            continue
        lengths = {r["tokens"] for r in data}
        assert len(lengths) == 1, (name, lengths)
        length = lengths.pop()
        first = baselines.setdefault(length, {sid: rs[0] for sid, rs in by_id.items()})
        checks[name]["tokens_per_request"] = length
        checks[name]["same_sample_set_as_baseline"] = set(first) == set(by_id)
        checks[name]["bitwise_equal_to_baseline"] = set(first) == set(by_id) and all(
            all(x == first[sid] for x in rs) for sid, rs in by_id.items()
        )
    queues = {}
    path = root / "queue.jsonl"
    if path.exists():
        for row in lines(path):
            queues.setdefault(row["tag"], []).append(row["queue_s"])
    return {
        "forwards": forward_summary,
        "captures": captures,
        "correctness": checks,
        "queue_mean_ms": {k: statistics.mean(v) * 1000 for k, v in queues.items()},
        "measurement_notes": [
            "Forward diagnostics synchronize CUDA and are distinct from unprofiled E2E.",
            "Forward rates count each logical token once across the eight-GPU group.",
            "Nested and overlapping group timers must not be added together.",
            "No-publication retains global auxiliary capture computation.",
        ],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("experiments", nargs="+")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = summarize(args.root, args.experiments)
    value = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(value)
    else:
        print(value, end="")
