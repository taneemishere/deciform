"""Live smoke test for the LLM-side backends, ~5 API calls total.

    LLM_API_KEY=$(cat nvapi_key) BASETEN_API_KEY=$(cat baseten_key) \
        python scripts/smoke_llm.py

1. elephant-1 decides the typed fields; a chat LLM generates `summary`.
2. LLMAsJevBackend answers two questions via first-token logprobs — same
   decision-model semantics, powered by a generative LLM.
"""

import json
import os
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from deciform import Deciform
from deciform.backends.llm import LLMAsJevBackend, OpenAIGenerativeBackend
from deciform.backends.systemone import SystemOneBackend
from deciform.questions import ChoiceQuestion, NoulQuestion

# key files live next to the repo root; env vars take precedence
_key_dir = Path(__file__).resolve().parent.parent
if not os.environ.get("BASETEN_API_KEY"):
    os.environ["BASETEN_API_KEY"] = (_key_dir / "baseten_key").read_text().strip()
if not os.environ.get("LLM_API_KEY"):
    os.environ["LLM_API_KEY"] = (_key_dir / "nvapi_key").read_text().strip()

# any OpenAI-compatible chat endpoint works here
LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
LLM_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
LLM_API_KEY = os.environ["LLM_API_KEY"]

STATE = {
    "merchant": "Delta Airlines",
    "amount_usd": 847.20,
    "memo": "SFO-JFK round trip, client visit",
    "cardholder": "Taneem J.",
    "date": "2026-10-18",
    "expense_policy": {
        "domestic_airfare_max_usd": 500,
        "requires_receipt_over_usd": 75,
    },
    "receipt_attached": True,
}


class ExpenseDecision(BaseModel):
    category: Literal["meals", "travel", "software", "office_supplies", "other"]
    policy_violation: bool
    flags: list[Literal["duplicate", "over_limit", "missing_receipt", "personal"]]
    summary: str  # free text -> generative fallback


def main() -> None:
    print("=== extract: decision model decides, chat LLM writes 'summary' ===")
    sf = Deciform(
        backend=SystemOneBackend(model=SystemOneBackend.ELEPHANT),
        fallback=OpenAIGenerativeBackend(
            base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=LLM_MODEL,
            requests_per_minute=30),
    )
    t0 = time.perf_counter()
    res = sf.extract(ExpenseDecision, STATE)
    print(json.dumps(res.value.model_dump(), indent=2))
    print(f"total latency: {res.total_latency_ms:.0f} ms (incl. throttle spacing)")
    for t in res.fields:
        print(f"  {t.path:18s} backend={t.backend:45s} escalated={t.escalated}")

    print("\n=== LLM-as-Jev: same LLM as a decision model (logprob readout) ===")
    jev = LLMAsJevBackend(
        base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=LLM_MODEL,
        requests_per_minute=30)
    qs = [
        ChoiceQuestion("cat", "What expense category applies?",
                       {"meals": "food", "travel": "airfare/hotel", "software": "SaaS",
                        "other": "none"}),
        NoulQuestion("viol", "Does this expense violate the stated policy?"),
    ]
    answers = jev.evaluate(STATE, qs)
    for name, a in answers.items():
        print(f"  {name}: value={a.value} probs={a.probabilities} raw_mode={a.raw.get('mode')}")
        if a.raw.get("letter_probs"):
            print(f"      letter_probs={a.raw['letter_probs']}")


if __name__ == "__main__":
    main()
