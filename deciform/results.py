"""Result types: normalized answers, per-field traces, and the assembled object.

Every backend returns answers in its own wire shape; ``Answer`` normalizes them.
``DecisionResult`` is what the caller gets: a validated object plus a full
trace of which backend answered which field, at what confidence and cost —
the artifacts that make decisions debuggable and eval-able.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Answer:
    """Normalized answer to a single typed question."""

    type: str  # "choice" | "noul" | "score"
    value: Any  # chosen option key | P(true) | weighted score
    probabilities: dict[str, float] | None = None  # option -> prob (choice/score)
    confidence: float | None = None  # how concentrated the distribution is
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def top_probability(self) -> float:
        """Confidence of the decided value; the number thresholds should key on."""
        if self.type == "noul":
            p = float(self.value)
            return p if p >= 0.5 else 1.0 - p
        if self.type == "multinoul" and self.probabilities:
            # A multi-select is only as confident as its closest member call.
            return min(p if p >= 0.5 else 1.0 - p for p in self.probabilities.values())
        if self.probabilities:
            return max(self.probabilities.values())
        return self.confidence or 0.0


@dataclass
class FieldTrace:
    """What happened to one schema field during extraction."""

    path: str  # dotted path, e.g. "address.country"
    questions: list[str]  # question names asked for this field
    backend: str  # which backend answered (or "generative:<model>")
    answer: Answer | None
    escalated: bool = False  # True if the field fell back to the generative backend
    low_confidence: bool = False  # accepted despite being under threshold (no fallback)
    skipped: bool = False  # gated question never fired (e.g. Optional absent)
    latency_ms: float = 0.0
    cost_usd: float | None = None


@dataclass
class DecisionResult:
    """An assembled, schema-validated object plus the audit trail."""

    value: Any  # validated Pydantic model instance or plain dict
    fields: list[FieldTrace] = field(default_factory=list)
    escalated: list[str] = field(default_factory=list)  # paths sent to fallback
    total_latency_ms: float = 0.0
    total_cost_usd: float | None = None
    # token accounting across all backend calls (None where the API doesn't report it)
    usage: dict[str, int] = field(default_factory=dict)
