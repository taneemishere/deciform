"""Real-endpoint wiring: an expense review against live backends.

Set env vars, then `python examples/expense_review.py`:

    SYSTEMONE_BASE_URL   any /v1/systemone endpoint
    SYSTEMONE_API_KEY    bearer for it
    SYSTEMONE_MODEL      e.g. the elephant decision model id
    LLM_BASE_URL         any OpenAI-compatible chat endpoint (/v1)
    LLM_API_KEY          bearer for it
    LLM_MODEL            a chat model id

The schema is a stand-in for any policy/audit object in a pipeline:
typed fields the decision model answers in one batched call, a free-text
field the generative fallback writes, and a fitted escalation threshold.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from deciform import Deciform, EscalationPolicy
from deciform.backends import OpenAIGenerativeBackend, SystemOneBackend


class ExpenseReview(BaseModel):
    category: Literal["meals", "travel", "software", "office_supplies", "other"]
    policy_violation: bool
    severity: Literal["none", "minor", "major"]
    flags: list[Literal["duplicate", "over_limit", "missing_receipt", "personal"]]
    reviewer: Literal["auto_approve", "manager", "finance_team"] | None = None
    notes: str = ""


RECEIPT = {
    "merchant": "Delta Airlines",
    "amount_usd": 847.20,
    "currency": "USD",
    "receipt_attached": False,
    "policy": {"domestic_airfare_max_usd": 500, "receipt_required_over_usd": 25},
    "submitted_by": "a.nguyen",
}


def main():
    sf = Deciform(
        backend=SystemOneBackend(
            base_url=os.environ["SYSTEMONE_BASE_URL"],
            api_key=os.environ["SYSTEMONE_API_KEY"],
            model=os.environ["SYSTEMONE_MODEL"],
        ),
        fallback=OpenAIGenerativeBackend(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
            model=os.environ["LLM_MODEL"],
        ),
        escalation=EscalationPolicy(threshold=0.8, debias_k=3),
    )
    res = sf.extract(ExpenseReview, state=RECEIPT)

    print(res.value.model_dump_json(indent=2))
    print("\nfield traces:")
    for t in res.fields:
        tag = "escalated" if t.escalated else ("skipped" if t.skipped else "kept")
        prob = t.answer.top_probability if t.answer else 0.0
        print(f"  {t.path:<18} {t.backend:<22} top={prob:.2f}  {tag}")
    print(f"\nescalated: {res.escalated}  latency: {res.total_latency_ms:.0f} ms")


if __name__ == "__main__":
    main()
