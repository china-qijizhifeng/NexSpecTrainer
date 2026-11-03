"""Prove compact placement/selection and benchmark the full consumer selection."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from compact_candidates import compact_scores, select_compact
from fused_indexer import fused_scores
from sglang.kernels.ops.attention.dsv4.candidate_blocks import topk_among_blocks
from validate_indexer import data, ms


def integration():
    """Exercise actual request planning, padded K, consume and pool-slot writes."""
    import fused_indexer
    from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
        DenseBlocksBackend,
    )

    q, keys, w, _ = data(9, 1042)
    lens = torch.tensor([0, 1025, 1, 17, 5, 1024, 0, 8, 16], device="cuda")
    slots = torch.randperm(1042, device="cuda")
    pool_keys = torch.empty_like(keys)
    pool_keys[slots] = keys
    raw = torch.empty((9, 40), device="cuda", dtype=torch.int32)
    pages = torch.empty_like(raw)
    inputs = SimpleNamespace(
        compress_ratio=4,
        req_rows=torch.tensor([0, 0, 1, 1, 0, 0, 1, 1, 1], device="cuda"),
        positions=(lens * 4 - 1).clamp_min(0),
        req_pool_indices=torch.tensor([0, 1], device="cuda"),
        seq_lens_cpu=[4100, 68],
        k_slots=slots,
        layer_id=0,
        q_lora=q,
        freqs_cis=torch.zeros((4100, 1), device="cuda"),
        x=w,
        indexer=SimpleNamespace(
            queries=lambda q_lora, freqs: q_lora,
            head_weights=lambda x: x,
            index_topk=40,
        ),
        out_raw_indices=raw,
        out_page_indices=pages,
        reset_outputs=lambda: (raw.fill_(-1), pages.fill_(-1)),
    )
    backend = SimpleNamespace(
        block_size=8,
        token_to_kv_pool=SimpleNamespace(
            get_low_ratio_index_k_dequant=lambda layer, selected: pool_keys[selected]
        ),
    )
    blocks = torch.tensor([128, 0, -1, 1], device="cuda", dtype=torch.int32)
    published = SimpleNamespace(blocks=blocks.repeat(9, 1))
    for ties in (False, True):
        if ties:
            w.zero_()
        fused_indexer.configure(True)
        DenseBlocksBackend._torch_consume_prefill(backend, inputs, published)
        expected = raw.clone(), pages.clone()
        fused_indexer.configure(True, compact=True)
        DenseBlocksBackend._torch_consume_prefill(backend, inputs, published)
        assert torch.equal(raw, expected[0]), "request-local selection mismatch"
        assert torch.equal(pages, expected[1]), "pool-slot mapping mismatch"
        assert fused_indexer._candidates is None
    fused_indexer.configure(False)
    return {"integration": "prefill_requests/consume/write_prefill", "exact": True}


def candidates(rows, keys, count):
    n = (keys + 7) // 8
    blocks = torch.stack(
        [torch.randperm(n, device="cuda")[:count] for _ in range(rows)]
    ).int()
    if rows:
        blocks[0] = -1
    return blocks.contiguous()


def compare(q, k, w, lens, blocks, topk, tile=128):
    dense = fused_scores(q, k, w, lens, blocks)
    compact = compact_scores(q, k, w, lens, blocks, keys_tile=tile)
    ids = (blocks[:, :, None].long() * 8 + torch.arange(8, device="cuda")).flatten(1)
    valid = (
        (blocks[:, :, None] >= 0).expand(-1, -1, 8).flatten(1)
        & (ids < lens[:, None])
        & (ids < k.shape[0])
    )
    want = dense.gather(1, ids.clamp(0, k.shape[0] - 1)).masked_fill(~valid, -torch.inf)
    assert torch.equal(compact, want), "compact slot or arithmetic mismatch"
    old = topk_among_blocks(dense, lens, blocks, topk, 8)
    new = select_compact(compact, blocks, topk)
    assert torch.equal(
        old.sort(-1).values, new.sort(-1).values
    ), "top-k mapping/tie mismatch"
    return new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(20261103)
    torch.set_num_threads(8)
    results = [integration()]
    print(json.dumps(results[-1]), flush=True)
    for rows, keys, count, topk, random in [
        (7, 17, 3, 17, False),
        (33, 1025, 64, 512, False),
        (17, 259, 3, 40, True),
        (65, 4096, 0, 512, False),
        (65, 4096, 513, 512, False),
        (256, 16384, 2048, 512, False),
        (256, 65536, 2048, 512, False),
    ]:
        q, k, w, lens = data(rows, keys, random)
        # Exercise the row strides accepted by the kernel, not just slice offsets.
        q_storage = torch.empty((rows * 2, 32, 128), device=q.device, dtype=q.dtype)
        w_storage = torch.empty((rows * 2, 32), device=w.device, dtype=w.dtype)
        q_storage[::2], w_storage[::2] = q, w
        q, w = q_storage[::2], w_storage[::2]
        blocks = candidates(rows, keys, count)
        for tie in [False, True]:
            if tie:
                w.zero_()
            selected = compare(q, k, w, lens, blocks, topk)
            # Same result with a tail query tile, and strided Q/weights views.
            pieces = []
            for i in range(0, rows, 19):
                pieces.append(
                    compare(
                        q[i : i + 19],
                        k,
                        w[i : i + 19],
                        lens[i : i + 19],
                        blocks[i : i + 19],
                        topk,
                    )
                )
            assert torch.equal(
                selected.sort(-1).values, torch.cat(pieces).sort(-1).values
            )
            results.append(
                dict(
                    rows=rows,
                    keys=keys,
                    blocks=blocks.shape[1],
                    topk=topk,
                    ties=tie,
                    exact=True,
                )
            )
            print(json.dumps(results[-1]), flush=True)
    for rows, keys in [(256, 16384), (256, 65536)]:
        q, k, w, lens = data(rows, keys)
        lens = torch.arange(keys - rows + 1, keys + 1, device="cuda")
        blocks = candidates(rows, keys, 2048)
        baseline = ms(
            lambda: topk_among_blocks(
                fused_scores(q, k, w, lens, blocks), lens, blocks, 512, 8
            )
        )
        for tile in [64, 128, 256]:
            compare(q, k, w, lens, blocks, 512, tile)
            timing = ms(
                lambda: select_compact(
                    compact_scores(q, k, w, lens, blocks, keys_tile=tile), blocks, 512
                )
            )
            result = dict(
                rows=rows,
                keys=keys,
                keys_tile=tile,
                baseline_ms=baseline,
                compact_ms=timing,
                speedup=baseline / timing,
                dense_score_bytes=rows * keys * 4,
                compact_score_bytes=rows * blocks.shape[1] * 8 * 4,
            )
            results.append(result)
            print(json.dumps(result), flush=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
