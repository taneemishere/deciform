"""Live smoke test: elephant-1 (and optionally mammoth-1) via SystemOneBackend.

Usage:
    BASETEN_API_KEY=$(cat baseten_key) python scripts/smoke_systemone.py [--mammoth]

Runs one raw evaluate() call and one Deciform.extract() end-to-end on an
expense-policy decision object.
"""

import argparse
import json
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field

from deciform import Deciform
from deciform.backends.systemone import SystemOneBackend

STATE = {
    "merchant": "Delta Airlines",
    "amount_usd": 847.20,
    "memo": "SFO-JFK round trip, client visit",
    "cardholder": "Taneem J.",
    "date": "2026-10-18",
    "expense_policy": {
        "domestic_airfare_max_usd": 500,
        "requires_receipt_over_usd": 75,
        "categories": ["meals", "travel", "software", "office_supplies", "other"],
    },
    "receipt_attached": True,
}


class ExpenseDecision(BaseModel):
    category: Literal["meals", "travel", "software", "office_supplies", "other"]
    policy_violation: bool
    severity: Literal["none", "minor", "major"]
    urgency: int = Field(ge=1, le=3, description="How urgent is review, 1=low 3=high")
    flags: list[Literal["duplicate", "over_limit", "missing_receipt", "personal"]]
    reviewer: Optional[Literal["auto_approve", "manager", "finance_team"]] = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mammoth", action="store_true", help="also run mammoth-1")
    args = parser.parse_args()

    for model in ([SystemOneBackend.ELEPHANT] + ([SystemOneBackend.MAMMOTH] if args.mammoth else [])):
        backend = SystemOneBackend(model=model)
        print(f"\n=== {backend.name} ===")

        # raw evaluate: three question types in one call
        from deciform.questions import ChoiceQuestion, NoulQuestion, ScoreQuestion

        t0 = time.perf_counter()
        raw = backend.evaluate(STATE, [
            ChoiceQuestion("cat", "What expense category applies?",
                           {"meals": "food", "travel": "airfare/hotel/transport",
                            "software": "SaaS/tools", "other": "none of the above"}),
            NoulQuestion("viol", "Does this expense violate the stated policy?"),
            ScoreQuestion("sev", "How severe is the violation?",
                          ["none", "minor", "major"]),
        ])
        dt = (time.perf_counter() - t0) * 1000
        print(f"raw evaluate ({dt:.0f} ms): "
              f"cat={raw['cat'].value} p={raw['cat'].probabilities}, "
              f"viol={raw['viol'].value:.3f}, sev={raw['sev'].value:.2f} "
              f"probs={raw['sev'].probabilities}")

        # end-to-end extract
        sf = Deciform(backend=backend)
        res = sf.extract(ExpenseDecision, STATE)
        print(f"extract ({res.total_latency_ms:.0f} ms): {json.dumps(res.value.model_dump(), indent=2)}")
        print("traces:")
        for t in res.fields:
            prob = t.answer.top_probability if t.answer else None
            print(f"  {t.path:20s} q={t.questions} prob={prob} skipped={t.skipped}")


if __name__ == "__main__":
    main()
