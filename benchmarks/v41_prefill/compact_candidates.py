"""Candidate-only score storage, bounded query tiles, unchanged torch top-k.

Nextron HyScale uses bounded query batches and threshold/filter/recovery to
avoid full score tables. Here the model already supplies at most 2048 blocks:
use their existing order as the compact column axis and retain torch's tie
handling. No sampled threshold, approximate truncation, or hash is needed.
"""

import torch
import triton
import triton.language as tl


def compact_scores(q, k, weights, lens, blocks, block_size=8, *, keys_tile=128):
    from fused_indexer import _score

    assert q.is_cuda and q.dtype == k.dtype == weights.dtype == torch.bfloat16
    assert q.shape[1:] == (32, 128) and k.shape[1] == 128
    assert q.stride(2) == 1 and q.stride(1) == 128 and k.stride(1) == 1
    assert weights.shape == q.shape[:2] and weights.stride(1) == 1
    assert lens.shape == (q.shape[0],) and lens.is_contiguous()
    assert blocks.shape[0] == q.shape[0] and blocks.is_contiguous()
    rows, width = q.shape[0], blocks.shape[1] * block_size
    out = torch.empty((rows, width), device=q.device, dtype=torch.float32)
    if rows and width:
        # Every compact slot is written, including invalid/empty candidate tiles.
        _score[(rows, triton.cdiv(width, keys_tile))](
            q,
            k,
            weights,
            lens,
            blocks,
            out,
            rows,
            k.shape[0],
            32,
            q.stride(0),
            k.stride(0),
            weights.stride(0),
            blocks.shape[1],
            block_size,
            True,
            1,
            keys_tile,
            COMPACT=True,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _map_selected(
    VALUES,
    PICKS,
    BLOCKS,
    OUT,
    K: tl.constexpr,
    PICK_K: tl.constexpr,
    NBLOCKS: tl.constexpr,
    BS: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, TILE)
    pick = tl.load(PICKS + row * PICK_K + col, col < PICK_K, 0)
    value = tl.load(VALUES + row * PICK_K + col, col < PICK_K, -float("inf"))
    block = tl.load(BLOCKS + row * NBLOCKS + pick // BS, col < PICK_K, -1)
    result = tl.where(
        (col < PICK_K) & (value != -float("inf")), block * BS + pick % BS, -1
    )
    tl.store(OUT + row * K + col, result, col < K)


def select_compact(scores, blocks, k, block_size=8):
    """Map unchanged top-k slots back to logical keys, padding missing keys -1."""
    rows, width = scores.shape
    if not width:
        return torch.full((rows, k), -1, device=scores.device, dtype=torch.int64)
    top = scores.topk(min(k, width), dim=-1, sorted=False)
    out = torch.empty((rows, k), device=scores.device, dtype=torch.int64)
    if rows and k:
        _map_selected[(rows,)](
            top.values,
            top.indices,
            blocks,
            out,
            k,
            top.values.shape[1],
            blocks.shape[1],
            block_size,
            triton.next_power_of_2(k),
        )
    return out


def consume_prefill(backend, inputs, published):
    from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import (
        prefill_requests,
        write_prefill,
    )

    for request, chunks in prefill_requests(
        inputs=inputs, token_to_kv_pool=backend.token_to_kv_pool
    ):
        for chunk in chunks:
            selected = select_compact(
                chunk.scores,
                published.blocks[chunk.tok].contiguous(),
                request.k,
                backend.block_size,
            )
            write_prefill(
                inputs, request, chunk, selected.masked_fill(selected < 0, request.lc)
            )
