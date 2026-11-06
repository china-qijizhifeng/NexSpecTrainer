"""GPU regression for the two shape-dependent arithmetic failures.

Use the real operands exported by resident_cp_kernel_regression (mHC) and
resident_cp_index_weights (indexer). No model weights need to be reloaded.
The real HTTP capture comparison additionally tests collectives and routing.
"""

import argparse
import json
from pathlib import Path

import torch

from ops.northgate_capture.cp_numerics import (
    combine_norm_prefill,
    gather_sequence,
    local_rows,
)


def check(index_fixture, mhc_fixture, stats_fixture=None):
    from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm

    index = torch.load(index_fixture, weights_only=True, map_location="cuda")
    x, weight = index["x"], index["weight"]
    reference = torch.nn.functional.linear(x, weight)
    sharded = torch.empty_like(reference)
    for rank in range(8):
        sharded[rank::8] = torch.nn.functional.linear(x[rank::8].contiguous(), weight)
    assert not torch.equal(
        reference, sharded
    ), "fixture no longer reproduces index GEMM drift"
    assert torch.equal(reference, index["full"]), "full-row projection changed"
    assert torch.equal(
        reference, torch.nn.functional.linear(x, weight)
    ), "projection is not repeatable"

    mhc = torch.load(mhc_fixture, weights_only=True, map_location="cuda")
    rows = torch.arange(4096, device="cuda") % len(mhc["x"])
    x, pre, weight, eps = mhc["x"][rows], mhc["pre"][rows], mhc["weight"], mhc["eps"]
    reference = hc_combine_norm(x.flatten(1), pre, weight, eps)
    old = hc_combine_norm(x[:512].flatten(1), pre[:512], weight, eps)
    assert not torch.equal(
        old, reference[:512]
    ), "fixture no longer reproduces mHC warp drift"
    for n in (1, 32, 96, 511, 512, 513, 4095, 4096):
        actual = combine_norm_prefill(x[:n], pre[:n], weight, eps)
        assert torch.equal(actual, reference[:n]), f"mHC depends on row count {n}"
    stats = check_stats(stats_fixture) if stats_fixture is not None else None
    return {
        "passed": True,
        "stats": stats,
        "mhc_row_counts": [1, 32, 96, 511, 512, 513, 4095, 4096],
        "original_index_differences": int((index["full"] != sharded).sum()),
        "original_mhc_differences": int((old != reference[:512]).sum()),
    }


def check_stats(path):
    from sglang.kernels.ops.layernorm.mhc import hc_mix_stats_sinkhorn_bf16x3

    from ops.northgate_capture.cp_numerics_kernels import mix_stats_bf16x3

    fixture = torch.load(path, weights_only=True, map_location="cuda")
    expected = fixture.pop("expected")
    n = expected[0].shape[0]
    old_mismatches = 0
    previous = None
    for _ in range(5):
        old = hc_mix_stats_sinkhorn_bf16x3(**fixture)
        old_mismatches += sum(int((a[:n] != b).sum()) for a, b in zip(old, expected))
        actual = mix_stats_bf16x3(**fixture)
        assert all(torch.equal(a[:n], b) for a, b in zip(actual, expected))
        if previous is not None:
            assert all(torch.equal(a, b) for a, b in zip(actual, previous))
        previous = actual
    assert old_mismatches > 0, "fixture no longer reproduces the 64-row tile failure"
    return {
        "five_repeats_exact": True,
        "reference_exact": True,
        "original_differences_across_five_runs": old_mismatches,
    }


def check_collectives(parallel):
    """Call on every live rank; exercise unequal logical counts on real NCCL."""
    from types import SimpleNamespace

    tested = (4095, 4096, 4097, 14226, 16383, 16384)
    for n in tested:
        rows = (n + parallel.attn_cp_size - 1) // parallel.attn_cp_size
        batch = SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(
                per_rank_actual_token=[rows] * parallel.attn_cp_size, total_seq_lens=n
            )
        )
        local = torch.arange(
            parallel.attn_cp_rank,
            n,
            parallel.attn_cp_size,
            device="cuda",
            dtype=torch.int64,
        )[:, None]
        full = gather_sequence(local, batch, parallel)
        assert torch.equal(full[:, 0], torch.arange(n, device="cuda"))
        assert torch.equal(
            local_rows(full, parallel.attn_cp_rank, parallel.attn_cp_size, len(local)),
            local,
        )
    return {"passed": True, "lengths": list(tested)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-fixture", type=Path, required=True)
    parser.add_argument("--mhc-fixture", type=Path, required=True)
    parser.add_argument("--stats-fixture", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            check(args.index_fixture, args.mhc_fixture, args.stats_fixture), indent=2
        )
    )
