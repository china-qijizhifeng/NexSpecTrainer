"""Tiled GPU suffix search and committed-history writes."""

import torch
import triton
import triton.language as tl

from .ngram import NgramResult


@triton.jit
def _scan_tiles(H, L, R, Scores, C: tl.constexpr, S: tl.constexpr,
                NT: tl.constexpr, MIN: tl.constexpr, MAX: tl.constexpr,
                BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    slot = tl.load(R + row)
    valid_slot = (slot >= 0) & (slot < S)
    size = tl.load(L + slot, valid_slot, other=0)
    valid_size = (size >= MIN + 1) & (size <= C)
    end = tile * BLOCK + tl.arange(0, BLOCK)
    live = valid_slot & valid_size & (end < size - 1)
    n = tl.full((BLOCK,), 0, tl.int32)
    for j in tl.static_range(MAX):
        a = tl.load(H + slot.to(tl.int64) * C + end - j,
                    live & (end >= j), other=-1)
        b = tl.load(H + slot.to(tl.int64) * C + size - 1 - j,
                    valid_slot & valid_size & (size > j), other=-2)
        same = live & (end >= j) & (n == j) & (a == b)
        n += same.to(tl.int32)
    score = tl.where(live & (n >= MIN), n.to(tl.int64) * (C + 1) + end + 1, -1)
    first = tl.max(score, 0)
    second = tl.max(tl.where(score == first, -1, score), 0)
    base = (row * NT + tile) * 2
    tl.store(Scores + base, first)
    tl.store(Scores + base + 1, second)


@triton.jit
def _scan_reduce(H, L, R, Scores, Tokens, Lengths, Match, Agree, Copy,
                 C: tl.constexpr, S: tl.constexpr, NT: tl.constexpr,
                 MAX: tl.constexpr, D: tl.constexpr, REDUCE: tl.constexpr,
                 D_P2: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(R + row)
    size = tl.load(L + slot, (slot >= 0) & (slot < S), other=0)
    i = tl.arange(0, REDUCE)
    scores = tl.load(Scores + row * NT * 2 + i, i < NT * 2, other=-1)
    first = tl.max(scores, 0)
    second = tl.max(tl.where(scores == first, -1, scores), 0)
    found = first >= 0
    n = tl.where(found, first // (C + 1), 0).to(tl.int32)
    end = first % (C + 1) - 1
    end2 = second % (C + 1) - 1
    count = tl.where(found, tl.minimum(D, size - end - 1), 0)
    j = tl.arange(0, D_P2)
    tok = tl.load(H + slot.to(tl.int64) * C + end + 1 + j,
                  found & (j < count), other=0)
    tl.store(Tokens + row * D + j, tok.to(tl.int64), j < D)
    next1 = tl.load(H + slot.to(tl.int64) * C + end + 1, found, other=0)
    next2 = tl.load(H + slot.to(tl.int64) * C + end2 + 1, second >= 0, other=0)
    agreement = (second < 0) | (next1 == next2)
    # Only the winning full match needs the more expensive backward extension.
    m = n
    if n == MAX:
        back = tl.arange(0, 256)
        a = tl.load(H + slot.to(tl.int64) * C + end - back,
                    found & (end >= back), other=-1)
        b = tl.load(H + slot.to(tl.int64) * C + size - 1 - back,
                    found & (size > back), other=-2)
        mismatch = (back > end) | (a != b)
        m = tl.min(tl.where(mismatch, back, 256), 0)
    tl.store(Lengths + row, count.to(tl.int32))
    tl.store(Match + row, n)
    tl.store(Agree + row, agreement.to(tl.int32))
    tl.store(Copy + row, m.to(tl.int32))


def scan_cuda(history, lengths, req_slots, min_match, max_match, max_draft):
    batch, capacity = req_slots.numel(), history.shape[1]
    tokens = torch.empty((batch, max_draft), device=history.device, dtype=torch.int64)
    features = [torch.empty(batch, device=history.device, dtype=torch.int32) for _ in range(4)]
    if batch:
        tiles = triton.cdiv(capacity, 256)
        scores = torch.empty((batch, tiles, 2), device=history.device, dtype=torch.int64)
        _scan_tiles[(batch, tiles)](history, lengths, req_slots, scores, capacity,
                                   history.shape[0], tiles, min_match, max_match, 256)
        _scan_reduce[(batch,)](history, lengths, req_slots, scores, tokens, *features,
                              capacity, history.shape[0], tiles, max_match, max_draft,
                              triton.next_power_of_2(tiles * 2),
                              triton.next_power_of_2(max_draft))
    return NgramResult(tokens, *features)


@triton.jit
def _write_history(H, L, R, P, T, N, C: tl.constexpr, S: tl.constexpr,
                   WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    slot, pos, count = tl.load(R + row), tl.load(P + row), tl.load(N + row)
    valid = ((slot >= 0) & (slot < S) & (pos >= 0) & (count >= 0)
             & (count <= WIDTH) & (pos + count <= C))
    j = tl.arange(0, BLOCK)
    token = tl.load(T + row * WIDTH + j, valid & (j < count) & (j < WIDTH), other=0)
    tl.store(H + slot.to(tl.int64) * C + pos + j, token,
             valid & (j < count) & (j < WIDTH))
    tl.store(L + slot, pos + count, valid)


def write_history(history, lengths, slots, positions, tokens, counts):
    """Write only confirmed verify inputs (anchor + accepted drafts), never bonus twice."""
    width = tokens.shape[1]
    if slots.numel():
        _write_history[(slots.numel(),)](history, lengths, slots, positions,
                                         tokens.contiguous(), counts,
                                         history.shape[1], history.shape[0], width,
                                         triton.next_power_of_2(width))
