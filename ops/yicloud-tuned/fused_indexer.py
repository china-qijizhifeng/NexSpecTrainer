"""Opt-in SM90 prefill scoring experiment for SGLang 38d9e4a3f.

Keep the reference BF16 rounding points (dot, weighted product, head sum).
Only final [rows, keys] scores reach HBM. Candidate consumers gather K before
the dot and never score keys outside their published blocks. Decode is untouched.
"""

import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _score(
    Q,
    K,
    W,
    LENS,
    BLOCKS,
    OUT,
    ROWS: tl.constexpr,
    KEYS: tl.constexpr,
    H: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    WS: tl.constexpr,
    NBLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CANDIDATE: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    COMPACT: tl.constexpr = False,
):
    rb, kb = tl.program_id(0), tl.program_id(1)
    rr = rb * R + tl.arange(0, R)
    lens = tl.load(LENS + rr, rr < ROWS, 0)
    col = kb * N + tl.arange(0, N)
    if CANDIDATE:
        # Candidate lists vary per query, so R=1 on this path.
        block = tl.load(
            BLOCKS + rb * NBLOCKS + col // BLOCK_SIZE, col // BLOCK_SIZE < NBLOCKS, -1
        )
        key = block * BLOCK_SIZE + col % BLOCK_SIZE
        valid = (block >= 0) & (key < tl.sum(lens, 0)) & (key < KEYS)
        busy = tl.sum(valid.to(tl.int32), 0) > 0
    else:
        key = col
        valid = key < KEYS
        busy = kb * N < tl.max(lens, 0)
    if busy:
        m = tl.arange(0, R * H)
        row = rb * R + m // H
        head = m % H
        d = tl.arange(0, 128)
        q = tl.load(
            Q + row[:, None] * QS + head[:, None] * 128 + d[None, :],
            row[:, None] < ROWS,
            0,
        )
        k = tl.load(K + key[None, :] * KS + d[:, None], valid[None, :], 0)
        dot = tl.dot(q, k).to(tl.bfloat16).to(tl.float32)
        w = tl.load(W + row * WS + head, row < ROWS, 0).to(tl.float32)
        weighted = (tl.maximum(dot, 0.0) * w[:, None]).to(tl.bfloat16).to(tl.float32)
        score = tl.sum(weighted.reshape(R, H, N), 1).to(tl.bfloat16).to(tl.float32)
        keep = valid[None, :] & (key[None, :] < lens[:, None])
        score = tl.where(keep, score, -float("inf"))
        if COMPACT:
            tl.store(
                OUT + rb * NBLOCKS * BLOCK_SIZE + col,
                score.reshape(N),
                col < NBLOCKS * BLOCK_SIZE,
            )
        else:
            tl.store(
                OUT + rr[:, None] * KEYS + key[None, :],
                score,
                (rr[:, None] < ROWS) & valid[None, :],
            )
    elif COMPACT:
        tl.store(
            OUT + rb * NBLOCKS * BLOCK_SIZE + col,
            -float("inf"),
            col < NBLOCKS * BLOCK_SIZE,
        )


def fused_scores(
    q, k, weights, lens, blocks=None, block_size=8, *, rows_tile=2, keys_tile=128
):
    """Dense-layout, causally masked scores; candidate-excluded entries are -inf."""
    assert q.is_cuda and q.dtype == k.dtype == weights.dtype == torch.bfloat16
    assert q.shape[1:] == (32, 128) and k.shape[1] == 128
    assert q.stride(2) == 1 and q.stride(1) == 128 and k.stride(1) == 1
    assert weights.shape == q.shape[:2] and weights.stride(1) == 1
    assert lens.shape == (q.shape[0],) and lens.is_contiguous()
    rows, keys = q.shape[0], k.shape[0]
    out = torch.full((rows, keys), -torch.inf, device=q.device, dtype=torch.float32)
    candidate = blocks is not None
    width = blocks.shape[1] * block_size if candidate else keys
    if rows == 0 or width == 0:
        return out
    if candidate:
        assert blocks.shape[0] == rows and blocks.is_contiguous()
        rows_tile = 1
    _score[(triton.cdiv(rows, rows_tile), triton.cdiv(width, keys_tile))](
        q,
        k,
        weights,
        lens,
        blocks if candidate else lens,
        out,
        rows,
        keys,
        32,
        q.stride(0),
        k.stride(0),
        weights.stride(0),
        blocks.shape[1] if candidate else 0,
        block_size,
        candidate,
        rows_tile,
        keys_tile,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


_original_chunks = None
_original_consume = None
_candidates = None
_enabled = False
_deepinfer = False
_compact = False
_audited = set()


def _audit(indexer, q, k, weights, lens, blocks, block_size, got):
    """One final-chunk sample per layer/length, in warmup only; never timed."""
    identity = (_deepinfer, _compact, id(indexer), k.shape[0])
    if not os.environ.get("SPECFORGE_INDEXER_AUDIT") or identity in _audited:
        return
    _audited.add(identity)
    from sglang.kernels.ops.attention.dsv4.candidate_blocks import topk_among_blocks
    from sglang.srt.runtime_context import get_parallel

    rows = torch.linspace(
        0, q.shape[0] - 1, min(32, q.shape[0]), device=q.device
    ).long()
    want = indexer.scores(q[rows], k, weights[rows])
    want.masked_fill_(
        torch.arange(k.shape[0], device=q.device)[None] >= lens[rows, None], -torch.inf
    )
    have = got[rows]
    width = min(indexer.index_topk, k.shape[0])
    if blocks is not None:
        left = topk_among_blocks(want, lens[rows], blocks[rows], width, block_size)
        if _compact:
            from compact_candidates import select_compact

            right = select_compact(have, blocks[rows].contiguous(), width, block_size)
            ids = (
                blocks[rows, :, None].long() * block_size
                + torch.arange(block_size, device=q.device)
            ).flatten(1)
            want = want.gather(1, ids.clamp(0, k.shape[0] - 1))
        else:
            right = topk_among_blocks(have, lens[rows], blocks[rows], width, block_size)
        valid = torch.isfinite(have)
    else:
        left = want.topk(width, sorted=False).indices
        right = have.topk(width, sorted=False).indices
        valid = torch.isfinite(want)
    assert torch.isfinite(have[valid]).all()
    x, y = want[valid], have[valid]
    record = {
        "rank": get_parallel().tp_rank,
        "keys": k.shape[0],
        "sample_rows": len(rows),
        "candidate": blocks is not None,
        "backend": (
            "compact"
            if _compact and blocks is not None
            else ("deepinfer" if _deepinfer and blocks is None else "triton")
        ),
        "score_exact_fraction": float((x == y).float().mean()),
        "max_abs": float((x - y).abs().max()),
        "relative_l2": float(
            torch.linalg.vector_norm(x - y)
            / torch.linalg.vector_norm(x).clamp_min(1e-30)
        ),
        "topk_exact": torch.equal(left.sort(-1).values, right.sort(-1).values),
    }
    with (
        Path(os.environ["PREFILL_ROOT"])
        / f"indexer-audit-rank{get_parallel().tp_rank}.jsonl"
    ).open("a") as f:
        f.write(json.dumps(record) + "\n")


def configure(enabled, *, deepinfer=False, compact=False):
    global _original_chunks, _original_consume, _enabled, _deepinfer, _compact
    from sglang.srt.layers.attention.dsv4.v41_indexer import dense_blocks, scoring

    if _original_chunks is None:
        _original_chunks = scoring._score_chunks
        _original_consume = dense_blocks.DenseBlocksBackend._torch_consume_prefill
        scoring._score_chunks = _score_chunks
        dense_blocks.DenseBlocksBackend._torch_consume_prefill = _consume
    _enabled = enabled
    _deepinfer = deepinfer
    _compact = compact


def deepinfer_scores(q, k, weights, lens):
    """Adapt DeepInfer's persistent keys-major scorer to the dense score contract."""
    from deepinfer_indexer import keys_major_logits, keys_major_supported

    q, k, weights = q.contiguous(), k.contiguous(), weights.contiguous()
    assert q.dtype == k.dtype == weights.dtype == torch.bfloat16
    assert keys_major_supported(q, k)
    out = torch.full(
        (q.shape[0], k.shape[0]), -torch.inf, device=q.device, dtype=torch.float32
    )
    starts = torch.zeros_like(lens, dtype=torch.int32)
    keys_major_logits(
        q, weights, k, starts, lens.to(torch.int32), out, k.shape[0], False
    )
    return out


def _consume(self, inputs, published):
    global _candidates
    previous = _candidates
    if _enabled and published is not None:
        _candidates = (published.blocks, self.block_size)
    try:
        if _enabled and _compact and published is not None:
            from compact_candidates import consume_prefill

            return consume_prefill(self, inputs, published)
        return _original_consume(self, inputs, published)
    finally:
        _candidates = previous


def _score_chunks(*, indexer, q, weights, index_k, request):
    from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import ChunkScores

    if not _enabled:
        yield from _original_chunks(
            indexer=indexer, q=q, weights=weights, index_k=index_k, request=request
        )
        return
    # Same row boundaries as reference, bounding final scores to 64 MiB and
    # preserving the shapes presented to the unchanged top-k implementation.
    rows_per_chunk = max(1, (1 << 30) // (q.shape[1] * index_k.shape[0] * 2))
    compact = _compact and _candidates is not None
    if compact:
        candidate_width = _candidates[0].shape[1] * _candidates[1]
        # Retain the original query batching when candidates span every key.
        # A tighter 16 MiB cap quadrupled launch/selection count at 16k and
        # regressed full prefill. At 64k the original row bound still gives a
        # 16 MiB compact score tensor (256 rows x 16384 candidate columns).
        rows_per_chunk = min(
            rows_per_chunk, max(1, (64 << 20) // max(4 * candidate_width, 1))
        )
    for start in range(0, request.tok.numel(), rows_per_chunk):
        rows = slice(start, start + rows_per_chunk)
        tok, lens = request.tok[rows], request.lens[rows]
        blocks = _candidates[0][tok].contiguous() if _candidates is not None else None
        score_fn = deepinfer_scores if _deepinfer and blocks is None else fused_scores
        if compact:
            from compact_candidates import compact_scores

            score_fn = compact_scores
        score_args = (
            ()
            if score_fn is deepinfer_scores
            else (blocks, _candidates[1] if blocks is not None else 8)
        )
        scores = score_fn(
            q[tok],
            index_k,
            weights[tok],
            lens,
            *score_args,
        )
        if start + rows_per_chunk >= request.tok.numel():
            _audit(
                indexer,
                q[tok],
                index_k,
                weights[tok],
                lens,
                blocks,
                _candidates[1] if blocks is not None else 8,
                scores,
            )
        yield ChunkScores(
            rows=rows,
            tok=tok,
            lens=lens,
            scores=scores if compact else scores[:, : request.lc],
        )
