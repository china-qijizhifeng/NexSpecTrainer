"""Conditional acceptance calibration, chain selection and masked prefix allocation."""

from dataclasses import dataclass

import torch


@dataclass
class ChainSelection:
    tokens: torch.Tensor
    survival: torch.Tensor
    lengths: torch.Tensor
    source: torch.Tensor  # 0=MTP, 1=n-gram, 2=contains extension


def estimator_features(result, max_match=15, min_match=5):
    """[match-first, match-later, agreement, full-first/later, logcopy-first/later]."""
    n = result.match_len.float()
    full = (n == max_match).float()
    logcopy = torch.log(result.copy_len.float().clamp(min=max_match) / max_match) * full
    return n - min_match, result.agreement.float(), full, logcopy


def estimate_ngram(result, coefficients):
    """Equation (1): beta[k], w[first/later], delta, ell[first/later], kappa[first/later]."""
    beta = coefficients["intercepts"]
    device = result.tokens.device
    beta = torch.as_tensor(beta, dtype=torch.float32, device=device)
    if beta.numel() != result.tokens.shape[1]:
        raise ValueError("calibrator horizon differs from n-gram horizon")
    n, agree, full, logcopy = estimator_features(result)
    first = torch.arange(beta.numel(), device=device) == 0
    def split(name):
        pair = torch.as_tensor(coefficients[name], dtype=torch.float32, device=device)
        return torch.where(first, pair[0], pair[1])
    logits = (beta[None, :] + n[:, None] * split("match")
              + agree[:, None] * coefficients["agreement"]
              + full[:, None] * split("full")
              + logcopy[:, None] * split("copy"))
    valid = torch.arange(beta.numel(), device=device)[None, :] < result.lengths[:, None]
    return torch.where(valid, logits.sigmoid(), 0.0)


def select_chains(mtp_tokens, mtp_conditional, ngram, ngram_conditional, *, slot_price=0.0,
                  contains=True, mode="hybrid"):
    """Select deterministic point-mass drafts before target verification.

    The contains merge takes max survival on the common prefix and applies
    n-gram conditional probabilities only after that prefix. Padded positions
    always have zero survival and token ID zero.
    """
    if mode not in ("hybrid", "mtp", "ngram"):
        raise ValueError("mode must be hybrid, mtp or ngram")
    batch, model_horizon = mtp_tokens.shape
    horizon = ngram.tokens.shape[1]
    if not 0 < model_horizon <= horizon or mtp_conditional.shape != mtp_tokens.shape:
        raise ValueError("MTP tokens/confidences must match and fit the verifier")
    if ngram_conditional.shape != ngram.tokens.shape:
        raise ValueError("n-gram probabilities must match tokens")
    mtp = torch.nn.functional.pad(mtp_tokens, (0, horizon - model_horizon))
    ms = torch.nn.functional.pad(mtp_conditional.float().cumprod(1), (0, horizon - model_horizon))
    ns = ngram_conditional.float().cumprod(1)
    positions = torch.arange(horizon, device=mtp.device)[None, :]
    ns = torch.where(positions < ngram.lengths[:, None], ns, 0.0)
    has_ngram = ngram.lengths > 0
    extension = (ngram.lengths > model_horizon) & (
        ngram.tokens[:, :model_horizon] == mtp_tokens).all(1)
    extension &= contains
    mv = (ms - slot_price).clamp(min=0).sum(1)
    nv = (ns - slot_price).clamp(min=0).sum(1)
    use_ngram = has_ngram & ((nv >= mv) | extension)
    if mode == "mtp":
        use_ngram = torch.zeros_like(use_ngram)
        extension = torch.zeros_like(extension)
    elif mode == "ngram":
        use_ngram = torch.ones_like(has_ngram)
        extension = torch.zeros_like(extension)
    prefix = torch.maximum(ms[:, :model_horizon], ns[:, :model_horizon])
    tail = prefix[:, -1:] * ngram_conditional[:, model_horizon:].float().cumprod(1)
    merged = torch.cat((prefix, tail), 1)
    selected_ns = torch.where(extension[:, None], merged, ns)
    survival = torch.where(use_ngram[:, None], selected_ns, ms)
    lengths = torch.where(use_ngram, ngram.lengths, model_horizon).to(torch.int32)
    valid = positions < lengths[:, None]
    tokens = torch.where(valid, torch.where(use_ngram[:, None], ngram.tokens, mtp), 0)
    return ChainSelection(tokens, torch.where(valid, survival, 0.0), lengths,
                          torch.where(use_ngram, torch.where(extension, 2, 1), 0).to(torch.int32))


def verify_lengths(survival, lengths, budget=None, survival_eps=1e-6):
    """Allocate extra verify slots by survival, preserving every request's prefix.

    Budget excludes the mandatory anchor. Stable position-then-request tie
    breaking makes tied probabilities prefix preserving. No token-dependent
    host transfer occurs, including when fewer valid slots exist than budget.
    """
    batch, horizon = survival.shape
    if budget is None:
        return lengths.to(torch.int32) + 1
    if type(budget) is not int or budget < 0:
        raise ValueError("extra-token budget must be a nonnegative host integer")
    pos = torch.arange(horizon, device=survival.device)[None, :]
    valid = (pos < lengths[:, None]) & (survival >= survival_eps)
    ranked = torch.where(valid, survival, -torch.inf).T.reshape(-1)
    order = torch.argsort(ranked, descending=True, stable=True)
    chosen = order[: min(budget, batch * horizon)]
    count = torch.zeros(batch, dtype=torch.int64, device=survival.device)
    count.scatter_add_(0, chosen % batch, valid.T.reshape(-1)[chosen].to(torch.int64))
    return (1 + count).to(torch.int32)
