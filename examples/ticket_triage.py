"""Offline demo: the full deciform pipeline with a scripted backend. No API keys.

`python examples/ticket_triage.py` walks one support ticket through the whole
stack — compile, batched decision round, Optional presence gate, confidence
escalation to a fallback, validated assembly — against a fake decision backend
that returns canned probabilities. Everything real except the HTTP.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from deciform import Deciform, EscalationPolicy
from deciform.questions import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion
from deciform.results import Answer


class TicketTriage(BaseModel):
    category: Literal["billing", "bug", "account", "feature_request", "other"]
    is_urgent: bool
    sentiment: Literal["angry", "neutral", "positive"]
    priority: int = Field(ge=1, le=5)
    needs_human: Literal["none", "support", "engineering"] | None = None
    reply_draft: str = ""


TICKET = {
    "subject": "Charged twice and your status page lied",
    "body": (
        "You billed me twice for March and the status page said 'no incidents' "
        "while checkout was down all morning. I need this fixed today."
    ),
    "customer_tier": "pro",
}


class ScriptedDecisionBackend:
    """DecisionBackend that returns canned probabilities per question name.

    Shows the contract any /v1/systemone backend satisfies: one evaluate()
    call answers every question in the batch with probabilities.
    """

    name = "scripted"

    def __init__(self):
        self.calls: list[list[str]] = []

    def evaluate(self, state: Any, questions: list[Question]) -> dict[str, Answer]:
        self.calls.append([q.name for q in questions])
        script = {
            "category": Answer(type="choice", value="billing", probabilities={
                "billing": 0.91, "bug": 0.05, "account": 0.02,
                "feature_request": 0.01, "other": 0.01}),
            "is_urgent": Answer(type="noul", value=0.94),
            "sentiment": Answer(type="choice", value="angry", probabilities={
                "angry": 0.96, "neutral": 0.03, "positive": 0.01}),
            "priority": Answer(type="score", value=3.0, probabilities={
                "0": 0.01, "1": 0.04, "2": 0.20, "3": 0.60, "4": 0.15}),
            "needs_human__present": Answer(type="noul", value=0.88),
            "needs_human": Answer(type="choice", value="support", probabilities={
                "none": 0.05, "support": 0.62, "engineering": 0.33}),
        }
        out = {}
        for q in questions:
            base = q.name.split("#rot")[0]
            if base in script:
                out[q.name] = script[base]
        return out


class ScriptedFallback:
    """GenerativeBackend that 'writes' the reply and covers escalated fields.

    Escalated typed fields come through here too — produce() must return a
    value conforming to the leaf's schema, which is what schema_hint is for.
    """

    name = "scriptedgen"

    def produce(self, state: Any, instructions: str, context: dict,
                schema_hint: dict | None = None) -> Any:
        if "reply_draft" in instructions:
            return ("Hi — you're right, you were billed twice and our status "
                    "page missed the outage. I've refunded the duplicate charge "
                    "and credited your account for the inconvenience.")
        if schema_hint and schema_hint.get("type") == "integer":
            return 4  # escalation target thinks this ticket is a 4/5
        return "engineering"


def main():
    backend = ScriptedDecisionBackend()
    sf = Deciform(
        backend=backend,
        fallback=ScriptedFallback(),
        escalation=EscalationPolicy(threshold=0.8),
    )
    res = sf.extract(TicketTriage, state=TICKET)

    print("== value ==")
    print(res.value.model_dump_json(indent=2))
    print("\n== rounds ==")
    for i, names in enumerate(backend.calls, 1):
        print(f"round {i}: {names}")
    print("\n== field traces ==")
    for t in res.fields:
        tag = "escalated" if t.escalated else ("skipped" if t.skipped else "kept")
        prob = t.answer.top_probability if t.answer else 0.0
        print(f"{t.path:<14} {t.backend:<18} top={prob:.2f}  {tag}")
    print(f"\nescalated: {res.escalated}")
    print(f"latency: {res.total_latency_ms:.0f} ms, usage: {res.usage}")


if __name__ == "__main__":
    main()
