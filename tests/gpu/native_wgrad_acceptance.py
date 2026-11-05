"""Compare native-size expert gradients and peak allocations with the old kernel."""

import argparse
import json
from pathlib import Path

import torch

from specforge.modeling.draft.deepseek_v41.kernels.moe import grouped_weight_gradient


def main():
    import deep_gemm

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    counts = [
        0 if i < 8 or i % 11 == 0 else alignment * (1 + i % 2) for i in range(128)
    ]
    lengths = torch.tensor(counts, device="cuda", dtype=torch.int32)
    records = []
    for outputs, inputs in ((4608, 5120), (5120, 2304)):
        x = torch.randn(sum(counts), inputs, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn(sum(counts), outputs, device="cuda", dtype=torch.bfloat16)
        weight = torch.empty(128, outputs, inputs, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        full = torch.zeros(weight.shape, device="cuda", dtype=torch.float32)
        deep_gemm.k_grouped_bf16_gemm_tn_contiguous(dy, x, full, counts, lengths, full)
        reference = full.to(weight.dtype)
        torch.cuda.synchronize()
        old_peak = torch.cuda.max_memory_allocated() - before
        del full
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        actual = grouped_weight_gradient(dy, x, weight, counts)
        torch.cuda.synchronize()
        new_peak = torch.cuda.max_memory_allocated() - before
        assert torch.equal(
            actual, reference
        ), "expert chunking changed native BF16 gradients"
        assert new_peak < old_peak * 0.5, (new_peak, old_peak)
        records.append(
            {
                "shape": list(weight.shape),
                "bitwise_equal": True,
                "baseline_peak_extra_bytes": old_peak,
                "chunked_peak_extra_bytes": new_peak,
            }
        )
        del x, dy, weight, reference, actual
    args.output.write_text(
        json.dumps({"passed": True, "cases": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
