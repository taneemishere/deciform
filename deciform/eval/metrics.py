"""Eval metrics for decision backends: accuracy, calibration, coverage, cost.

A FieldRecord is one decided field vs. its gold label — the atom every
metric and the threshold-fit routine works on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FieldRecord:
    path: str
    value: Any  # decided value
    label: Any  # gold
    probabilities: dict[str, float] | None  # option -> prob (noul: {"yes","no"})
    top_probability: float
    correct: bool
    skipped: bool = False
    escalated: bool = False
    latency_ms: float = 0.0


def accuracy(records: list[FieldRecord]) -> float:
    """Skipped fields count: abstention is right iff the label is null.
    The runner sets value=None on skipped fields, so val==label does this."""
    if not records:
        return 0.0
    return sum(r.correct for r in records) / len(records)


def ece(records: list[FieldRecord], n_bins: int = 10) -> float:
    """Expected calibration error: |confidence - accuracy| per bin, weighted."""
    recs = [r for r in records if not r.skipped]
    if not recs:
        return 0.0
    e = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        bin_ = [r for r in recs if lo <= r.top_probability < hi or (b == n_bins - 1 and r.top_probability == hi)]
        if not bin_:
            continue
        conf = sum(r.top_probability for r in bin_) / len(bin_)
        acc = sum(r.correct for r in bin_) / len(bin_)
        e += len(bin_) / len(recs) * abs(conf - acc)
    return e


def _gold_hit(label: Any, opt: str) -> float:
    if isinstance(label, list):
        return 1.0 if opt in {str(v) for v in label} else 0.0
    if isinstance(label, bool):
        return 1.0 if opt == ("yes" if label else "no") else 0.0
    return 1.0 if opt == str(label) else 0.0


def brier(records: list[FieldRecord]) -> float:
    """Mean Brier score over the decision distribution."""
    recs = [r for r in records if not r.skipped and r.probabilities]
    if not recs:
        return 0.0
    total = 0.0
    for r in recs:
        total += sum((p - _gold_hit(r.label, opt)) ** 2
                     for opt, p in r.probabilities.items())
    return total / len(recs)


def coverage_curve(records: list[FieldRecord], points: int = 51) -> list[dict]:
    """Selective-prediction curve: coverage + accepted error at each threshold."""
    recs = [r for r in records if not r.skipped]
    if not recs:
        return []
    taus = sorted({r.top_probability for r in recs})
    grid = [taus[0] + (taus[-1] - taus[0]) * i / (points - 1) for i in range(points)]
    out = []
    for tau in grid:
        acc = [r.correct for r in recs if r.top_probability >= tau]
        out.append({
            "tau": tau,
            "coverage": len(acc) / len(recs),
            "error": 1.0 - (sum(acc) / len(acc) if acc else 1.0),
            "n_accepted": len(acc),
        })
    return out


def auarc(records: list[FieldRecord]) -> float:
    """Area under the accuracy-vs-coverage (risk-coverage) curve. Higher = better."""
    recs = sorted((r for r in records if not r.skipped),
                  key=lambda r: r.top_probability, reverse=True)
    if not recs:
        return 0.0
    area, correct = 0.0, 0
    for i, r in enumerate(recs, 1):
        correct += r.correct
        area += correct / i  # trapezoid over coverage steps
    return area / len(recs)


def latency_stats(latencies: list[float]) -> dict[str, float]:
    ms = sorted(m for m in latencies if m > 0)
    if not ms:
        return {"p50": 0.0, "p95": 0.0}
    return {
        "p50": ms[len(ms) // 2],
        "p95": ms[min(len(ms) - 1, int(len(ms) * 0.95))],
    }
