"""Differential and representative-shape microbenchmarks for the Hopper port."""

import argparse
import json
from pathlib import Path

import torch
from fused_indexer import deepinfer_scores, fused_scores
from validate_indexer import data, ms, ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20261103)
    torch.set_num_threads(8)
    results = []
    cases = [(7, 17, False), (33, 1025, False), (65, 4096, False), (17, 259, True)]
    # Match score scratch row counts at 16k/64k, low and late causal ranges.
    cases += [
        (1024, 8192, False),
        (512, 16384, False),
        (512, 32768, False),
        (256, 65536, False),
    ]
    for rows, keys, random in cases:
        q, k, w, lens = data(rows, keys, random)
        for late in [False, True] if rows >= 256 else [False]:
            if late:
                lens = torch.arange(keys - rows + 1, keys + 1, device="cuda")
            got = deepinfer_scores(q, k, w, lens)
            # Torch reference bounded by row batches only to avoid its huge scratch.
            expected = torch.cat(
                [
                    ref(q[i : i + 16], k, w[i : i + 16], lens[i : i + 16])
                    for i in range(0, rows, 16)
                ]
            )
            valid = torch.isfinite(expected)
            assert torch.equal(valid, torch.isfinite(got))
            delta = got[valid] - expected[valid]
            row = dict(
                rows=rows,
                keys=keys,
                random_bf16=random,
                late=late,
                scores_exact=torch.equal(got, expected),
                max_abs=float(delta.abs().max()),
                relative_l2=float(
                    delta.norm() / expected[valid].norm().clamp_min(1e-30)
                ),
                topk_exact=torch.equal(
                    got.topk(min(512, keys), sorted=False).indices.sort(-1).values,
                    expected.topk(min(512, keys), sorted=False).indices.sort(-1).values,
                ),
            )
            torch.testing.assert_close(got, expected, rtol=0.01, atol=0.5)
            if not random:
                # DeepInfer sums heads sequentially, unlike torch's tree sum.
                # A CPU FP64 oracle isolated two BF16 boundary differences in
                # 7,864,832 scores. Preserve strict selection acceptance and
                # bound score error; full-model outputs are checked separately.
                assert row["relative_l2"] <= 1e-5 and row["topk_exact"], row
            row["current_ms"] = ms(lambda: fused_scores(q, k, w, lens))
            row["deepinfer_ms"] = ms(lambda: deepinfer_scores(q, k, w, lens))
            row["speedup"] = row["current_ms"] / row["deepinfer_ms"]
            results.append(row)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
