"""Real GPU differential at small, overflow-boundary and full trajectory shapes.

Reference launches the pinned original kernel on safe, independent row tiles.
All row arithmetic and reduction order stay identical; full-size outputs must
match bitwise, including every packed aux slot and the non-multiple tail.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plugins/sglang-spec-capture"))
from specforge_sglang_capture import mhc_index64
from sglang.kernels.ops.layernorm import mhc

p = argparse.ArgumentParser()
p.add_argument("--baseline", action="store_true")
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
torch.manual_seed(20261105)
torch.set_grad_enabled(False)
original_combine = mhc._mhc_combine_kernel
original_mix = mhc._hc_mix_stats_partial_kernel
if not args.baseline:
    mhc_index64.install()
results = []
for n in ([234048] if args.baseline else [8193, 104858, 234048]):
    x = torch.empty((n, 20480), dtype=torch.bfloat16, device="cuda")
    # Fill in safe tiles, with distinct values per row and channel.
    for start in range(0, n, 4096):
        x[start:start + 4096].normal_()
    pre = torch.randn((n, 4), dtype=torch.float32, device="cuda")
    got = mhc.mhc_combine(x, pre, 4, torch.bfloat16)
    torch.cuda.synchronize()
    # The original unsafe full-size kernel can fail or silently read wrong data.
    mhc._mhc_combine_kernel = original_combine
    for start in range(0, n, 4096):
        want = mhc.mhc_combine(x[start:start + 4096], pre[start:start + 4096], 4, torch.bfloat16)
        assert torch.equal(got[start:start + 4096], want), ("combine", n, start)
    mhc_index64.install()
    del got, pre

    # Existing mean producer already widens row offsets: audit every slot.
    packed = torch.empty((n, 3 * 5120), dtype=torch.bfloat16, device="cuda")
    for slot in range(3):
        out = packed[:, slot * 5120:(slot + 1) * 5120]
        mhc.mhc_stream_mean(x.view(n, 4, 5120), out=out)
        torch.cuda.synchronize()
        for start in range(0, n, 4096):
            want = mhc.mhc_stream_mean(x[start:start + 4096].view(-1, 4, 5120))
            assert torch.equal(out[start:start + 4096], want), ("aux", n, slot, start)
    del packed
    # Batch-invariant stats fallback has the same wide-row multiplication.
    weight = torch.randn((24, 20480), dtype=torch.float32, device="cuda") * 0.001
    got = mhc.hc_mix_stats(x, weight, 1e-6)
    torch.cuda.synchronize()
    mhc._hc_mix_stats_partial_kernel = original_mix
    for start in range(0, n, 4096):
        want = mhc.hc_mix_stats(x[start:start + 4096], weight, 1e-6)
        assert torch.equal(got[start:start + 4096], want), ("stats", n, start)
    mhc_index64.install()
    del x, weight, got
    results.append({"tokens": n, "combine_bitwise": True, "stats_bitwise": True,
                    "all_aux_slots_bitwise": True})
    print(json.dumps(results[-1]), flush=True)
args.output.write_text(json.dumps({"passed": True, "cases": results}, indent=2))
