"""Exact additive evaluation summaries with explicit populations and empty bins."""

from __future__ import annotations

import bisect
import math


def parse_bins(value):
    parts = value.split(",") if isinstance(value, str) else value
    edges = []
    for part in parts:
        text = str(part).strip().lower()
        number = (
            math.inf
            if text == "inf"
            else float(text[:-1]) * 1024 if text.endswith("k") else float(text)
        )
        if (
            math.isnan(number)
            or number < 0
            or (math.isfinite(number) and not number.is_integer())
        ):
            raise ValueError("ctx bin edges must be non-negative integers or inf")
        edges.append(number)
    if (
        len(edges) < 2
        or edges[0] != 0
        or edges[-1] != math.inf
        or any(a >= b for a, b in zip(edges, edges[1:]))
    ):
        raise ValueError("ctx bins must increase strictly from 0 to inf")
    return edges


class MetricTotals:
    def __init__(self, metrics, edges=None):
        self.metrics = tuple(metrics)
        self.edges = edges
        self.rows = {"overall": self._empty()}
        if edges:
            for index in range(len(edges) - 1):
                self.rows[str(index)] = self._empty()

    def _empty(self):
        return {name: [0.0, 0.0] for name in self.metrics}

    def add(self, terms, *, ctx_len=None, bin_index=None):
        if set(terms) != set(self.metrics):
            raise ValueError(
                f"metric terms do not match selection: {set(terms)} != {set(self.metrics)}"
            )
        targets = [self.rows["overall"]]
        if self.edges:
            if bin_index is None:
                if ctx_len is None or not math.isfinite(ctx_len) or ctx_len < 0:
                    raise ValueError("ctx_len must be finite and non-negative")
                bin_index = bisect.bisect_right(self.edges, ctx_len) - 1
            targets.append(self.rows[str(bin_index)])
        for metric, (numerator, denominator) in terms.items():
            numerator, denominator = float(numerator), float(denominator)
            if (
                not math.isfinite(numerator)
                or not math.isfinite(denominator)
                or denominator < 0
            ):
                raise ValueError(f"invalid additive terms for {metric}")
            if denominator == 0 and numerator != 0:
                raise ValueError(
                    f"nonzero numerator with zero denominator for {metric}"
                )
            for row in targets:
                row[metric][0] += numerator
                row[metric][1] += denominator

    def result(self):
        def render(row):
            return {
                name: {"value": n / d if d else None, "numerator": n, "denominator": d}
                for name, (n, d) in row.items()
            }

        result = {"overall": render(self.rows["overall"]), "bins": []}
        if self.edges:
            for index, (lo, hi) in enumerate(zip(self.edges, self.edges[1:])):
                result["bins"].append(
                    {
                        "lower": int(lo),
                        "upper": int(hi) if math.isfinite(hi) else None,
                        "metrics": render(self.rows[str(index)]),
                    }
                )
        return result
