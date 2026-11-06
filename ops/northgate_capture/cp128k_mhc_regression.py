"""Real-operand GPU proof of the 64K mHC dispatch discontinuity."""

import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

from ops.northgate_capture import mhc_index64
from ops.northgate_capture.cp_numerics_kernels import mix_stats_bf16x3

ROOT = Path(os.environ["RUN_DIR"])
DATA = Path(os.environ["NORTHGATE_ROOT"])
FIXTURE = (
    DATA
    / "runs/northgate-resident-tp-cp-16h200-20261106-r7/worker-1/control/commands/099-mhc-stats-fixture/mhc-stats-fixture.pt"
)
MODEL = Path("/gpfs/shared-model-weight/models/deepseek-ai/DeepSeek-V4.1-Flash-1017")


def long_stats(x, fn, fixture):
    from sglang.kernels.ops.layernorm import mhc

    mixes = mhc.hc_mix_stats(x, fn, fixture["rms_eps"]).unsqueeze(1)
    values = mhc.hc_split_sinkhorn(
        mixes,
        fixture["scale"],
        fixture["base"],
        4,
        fixture["sinkhorn_iters"],
        fixture["hc_eps"],
    )
    return tuple(v.squeeze(1) for v in values)


def diff(a, b):
    return [
        {"different": int((x != y).sum()), "max_abs": float((x - y).abs().max())}
        for x, y in zip(a, b)
    ]


def main():
    from sglang.kernels.ops.layernorm import mhc

    torch.set_num_threads(4)
    assert torch.cuda.get_device_capability() == (9, 0)
    mhc_index64.install()
    f = torch.load(FIXTURE, weights_only=True, map_location="cuda")
    f.pop("expected")
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    name = "layers.0.hc_attn_fn"
    with safe_open(MODEL / index[name], framework="pt", device="cpu") as source:
        fn = source.get_tensor(name).float().cuda().contiguous()
    parts = mhc.split_bf16_hc_weight(fn)
    assert all(
        torch.equal(a, b) for a, b in zip(parts, f["weight_parts"])
    ), "fixture weight mismatch"
    x = f.pop("x")
    records = []
    for n in [65537, 131072]:
        full = x[torch.arange(n, device="cuda") % len(x)].contiguous()
        ref = long_stats(full, fn, f)
        assert all(torch.equal(a, b) for a, b in zip(ref, long_stats(full, fn, f)))
        shards = []
        for rank in range(8):
            local = full[rank::8].contiguous()
            expected = tuple(v[rank::8] for v in ref)
            old = mix_stats_bf16x3(local, **f)
            fixed = long_stats(local, fn, f)
            again = long_stats(local, fn, f)
            old_diff = diff(old, expected)
            assert all(torch.equal(a, b) for a, b in zip(fixed, expected)), (
                n,
                rank,
                "corrected mismatch",
            )
            assert all(torch.equal(a, b) for a, b in zip(fixed, again)), (
                n,
                rank,
                "corrected repeat",
            )
            shards.append(
                {
                    "rank": rank,
                    "rows": len(local),
                    "old": old_diff,
                    "fixed_exact": True,
                    "repeat_exact": True,
                }
            )
        assert any(d["different"] for s in shards for d in s["old"])
        records.append(
            {"global_tokens": n, "full_repeat_exact": True, "shards": shards}
        )
        del full, ref
    result = {
        "passed": True,
        "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "weight": name,
        "weight_sha256": hashlib.sha256(fn.cpu().numpy().tobytes()).hexdigest(),
        "scope": "real 1784-row captured operands cyclically extended; not full-model 128K proof",
        "baseline": "hc_mix_stats + hc_split_sinkhorn",
        "old_cp": "BF16x3 128-row tile",
        "records": records,
    }
    (ROOT / "RESULT.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
