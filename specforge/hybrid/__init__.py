"""Request-local Hybrid Spark drafting, independent of a serving scheduler."""

__all__ = [
    "NgramResult", "scan", "ChainSelection", "estimate_ngram", "select_chains",
    "verify_lengths",
]


def __getattr__(name):
    # Configuration and patch preflight can run without importing torch.
    if name in ("NgramResult", "scan"):
        from . import ngram

        return getattr(ngram, name)
    if name in __all__:
        from . import selection

        return getattr(selection, name)
    raise AttributeError(name)
