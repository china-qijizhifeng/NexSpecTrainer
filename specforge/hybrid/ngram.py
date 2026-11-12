"""Exact suffix lookup over visible token IDs. CUDA dispatch never reads a device scalar."""

from dataclasses import dataclass

import torch


@dataclass
class NgramResult:
    tokens: torch.Tensor
    lengths: torch.Tensor
    match_len: torch.Tensor
    agreement: torch.Tensor
    copy_len: torch.Tensor


def scan(history, lengths, req_slots, *, min_match=5, max_match=15, max_draft=15):
    """Find the longest, then most recent historical suffix match per request.

    ``lengths`` is indexed by pool slot; visible history includes the current
    anchor exactly once. Copies may overlap the matching suffix but never read
    beyond the visible length. Invalid slots/lengths produce empty rows.
    """
    if not (1 <= min_match <= max_match <= 256 and 1 <= max_draft <= 256):
        raise ValueError("require 1 <= min_match <= max_match <= 256, max_draft in [1,256]")
    if history.ndim != 2 or lengths.ndim != 1 or req_slots.ndim != 1:
        raise ValueError("expected history[slots,context], lengths[slots], req_slots[batch]")
    if lengths.numel() != history.shape[0] or history.shape[1] < 1:
        raise ValueError("history and slot lengths disagree or context capacity is zero")
    if any(t.device != history.device for t in (lengths, req_slots)):
        raise ValueError("all n-gram inputs must be on the same device")
    if any(t.dtype not in (torch.int32, torch.int64) for t in (history, lengths, req_slots)):
        raise ValueError("token IDs, lengths and slots must be integer tensors")
    if not all(t.is_contiguous() for t in (history, lengths, req_slots)):
        raise ValueError("n-gram inputs must be contiguous")
    if history.is_cuda:
        from .kernels import scan_cuda

        return scan_cuda(history, lengths, req_slots, min_match, max_match, max_draft)
    if history.device.type != "cpu":
        raise ValueError("Hybrid Spark supports CPU reference and CUDA")
    return scan_reference(history, lengths, req_slots, min_match, max_match, max_draft)


def scan_reference(history, lengths, req_slots, min_match=5, max_match=15, max_draft=15):
    """Slow, explicit reference used for CPU replay and differential acceptance."""
    batch = req_slots.numel()
    tokens = torch.zeros((batch, max_draft), dtype=torch.int64)
    count = torch.zeros(batch, dtype=torch.int32)
    match = torch.zeros_like(count)
    agree = torch.ones_like(count)
    copy = torch.zeros_like(count)
    for row, slot in enumerate(req_slots.tolist()):
        if not 0 <= slot < history.shape[0]:
            continue
        size = int(lengths[slot])
        if not 0 <= size <= history.shape[1]:
            continue
        visible = history[slot, :size].tolist()
        matches = []
        for end in range(size - 1):
            n = 0
            while n < min(max_match, end + 1) and visible[end - n] == visible[size - 1 - n]:
                n += 1
            if n >= min_match:
                matches.append((n, end))
        matches.sort(reverse=True)
        if not matches:
            continue
        n, end = matches[0]
        k = min(max_draft, size - end - 1)
        tokens[row, :k] = torch.tensor(visible[end + 1 : end + 1 + k])
        count[row], match[row] = k, n
        if len(matches) > 1:
            agree[row] = int(visible[matches[1][1] + 1] == visible[end + 1])
        m = n
        if n == max_match:
            while m < min(256, end + 1) and visible[end - m] == visible[size - 1 - m]:
                m += 1
        copy[row] = m
    return NgramResult(tokens, count, match, agree, copy)
