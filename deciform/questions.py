"""Typed System One question primitives — the atoms a schema compiles into.

These mirror the three question types the System One API (and Jev-style
decision models generally) understands: choice, noul (yes/no), and score
(ordinal rubric). A ``deciform`` compile step turns each field of a
schema into one or more of these questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union


@dataclass(frozen=True)
class ChoiceQuestion:
    """Pick exactly one option from a named set (max options is backend-dependent)."""

    name: str
    instructions: str
    criteria: dict[str, str]  # option key -> human description
    type: str = "choice"


@dataclass(frozen=True)
class NoulQuestion:
    """Yes/no question. The answer is P(true) in [0, 1]."""

    name: str
    instructions: str
    type: str = "noul"


@dataclass(frozen=True)
class ScoreQuestion:
    """Rate the state against ordered rubric levels (2–10 depending on backend)."""

    name: str
    instructions: str
    criteria: list[str]  # ordered level descriptions, index = level
    type: str = "score"


Question = Union[ChoiceQuestion, NoulQuestion, ScoreQuestion]


def to_wire(question: Question) -> dict[str, Any]:
    """Serialize a question to the System One API request shape."""
    body: dict[str, Any] = {"type": question.type, "instructions": question.instructions}
    if isinstance(question, (ChoiceQuestion, ScoreQuestion)):
        body["criteria"] = question.criteria
    return body
