"""Real H100 differential scoring, candidate selection and timing; no model load."""

import argparse
import json
from pathlib import Path

import torch
from fused_indexer import fused_scores
from sglang.kernels.ops.attention.dsv4.candidate_blocks import topk_among_blocks
from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer


def fp4(shape):
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    x = table[torch.randint(8, shape, device="cuda")] * (
        torch.randint(2, shape, device="cuda") * 2 - 1
    )
    scale = 2.0 ** torch.randint(-3, 3, (*shape[:-1], shape[-1] // 32), device="cuda")
    return (x * scale.repeat_interleave(32, -1)).bfloat16()


def data(rows, keys, random=False):
    q = fp4((rows, 32, 128))
    k = fp4((keys, 128))
    w = torch.randn((rows, 32), device="cuda").bfloat16()
    if random:
        q = torch.randn_like(q)
        k = torch.randn_like(k)
    lens = torch.linspace(0, keys, rows, device="cuda").long()
    return q, k, w, lens


def ref(q, k, w, lens):
    scores = DeepseekV41Indexer.scores(None, q, k, w)
    return scores.masked_fill(
        torch.arange(k.shape[0], device="cuda")[None, :] >= lens[:, None], -torch.inf
    )


def ms(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(5):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / 5


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("indexer-validation.json"))
    p.add_argument("--tune", action="store_true")
    args = p.parse_args()
    torch.manual_seed(20261103)
    torch.set_num_threads(8)
    results = []
    for rows, keys, random in [
        (7, 17, False),
        (33, 1025, False),
        (65, 4096, False),
        (17, 259, True),
    ]:
        q, k, w, lens = data(rows, keys, random)
        expected = ref(q, k, w, lens)
        for candidate in [False, True]:
            blocks = None
            if candidate:
                nblocks = (keys + 7) // 8
                blocks = torch.stack(
                    [
                        torch.randperm(nblocks, device="cuda")[: min(nblocks, 64)]
                        for _ in range(rows)
                    ]
                ).int()
                blocks[0] = -1
            got = fused_scores(q, k, w, lens, blocks)
            if candidate:
                offsets = torch.arange(8, device="cuda")
                ids = (blocks[:, :, None].long() * 8 + offsets).flatten(1)
                selected = (blocks[:, :, None] >= 0).expand(-1, -1, 8).flatten(1) & (
                    ids < lens[:, None]
                )
                want = expected.gather(1, ids.clamp(0, keys - 1)).masked_fill(
                    ~selected, -torch.inf
                )
                have = got.gather(1, ids.clamp(0, keys - 1)).masked_fill(
                    ~selected, -torch.inf
                )
                left = (
                    topk_among_blocks(expected, lens, blocks, min(512, keys), 8)
                    .sort(-1)
                    .values
                )
                right = (
                    topk_among_blocks(got, lens, blocks, min(512, keys), 8)
                    .sort(-1)
                    .values
                )
            else:
                want, have = expected, got
                left = want.topk(min(512, keys), sorted=False).indices.sort(-1).values
                right = have.topk(min(512, keys), sorted=False).indices.sort(-1).values
            assert torch.equal(torch.isfinite(want), torch.isfinite(have))
            valid = torch.isfinite(want)
            delta = (want[valid] - have[valid]).abs()
            row = {
                "rows": rows,
                "keys": keys,
                "random_bf16": random,
                "candidate": candidate,
                "scores_exact": torch.equal(want, have),
                "max_abs": float(delta.max()) if delta.numel() else 0,
                "topk_exact": torch.equal(left, right),
            }
            results.append(row)
            print(json.dumps(row), flush=True)
            # FP4 inputs should reproduce reference arithmetic exactly.
            if not random:
                assert row["scores_exact"] and row["topk_exact"], row
            else:
                torch.testing.assert_close(have, want, rtol=0.01, atol=0.5)

    for rows, keys in [(512, 8192), (256, 65536)] if args.tune else []:
        q, k, w, lens = data(rows, keys)
        baseline_ms = ms(lambda: ref(q, k, w, lens))
        configs = []
        for rt in [1, 2, 4, 8]:
            for kt in [64, 128, 256]:
                configs.append(
                    {
                        "rows_tile": rt,
                        "keys_tile": kt,
                        "ms": ms(
                            lambda: fused_scores(
                                q, k, w, lens, rows_tile=rt, keys_tile=kt
                            )
                        ),
                    }
                )
        nblocks = min((keys + 7) // 8, 2048)
        blocks = (
            torch.arange(nblocks, device="cuda")
            .int()[None, :]
            .expand(rows, -1)
            .contiguous()
        )
        candidates = [
            {
                "keys_tile": kt,
                "ms": ms(lambda: fused_scores(q, k, w, lens, blocks, keys_tile=kt)),
            }
            for kt in [64, 128, 256]
        ]
        row = {
            "rows": rows,
            "keys": keys,
            "baseline_ms": baseline_ms,
            "dense": configs,
            "candidate": candidates,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
