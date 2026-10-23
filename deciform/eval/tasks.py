"""Eval tasks: each is a labeled decision dataset expressed as a Pydantic model.

Loaders:
- banking77: 77-way intent classification (choice stress test, PolyAI)
- sst5:      5-level sentiment rating (score primitive)
- rte:       textual entailment (noul primitive)
- fintech:   procedurally generated expense states with deterministic labels —
             the multi-field schema (incl. Optional gating) used end-to-end
"""

from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel, Field, create_model


@dataclass
class EvalExample:
    state: Any
    labels: dict[str, Any]  # field path -> gold value


@dataclass
class EvalTask:
    name: str
    model: type[BaseModel]
    examples: list[EvalExample]
    description: str = ""


def _literal_model(name: str, field_name: str, values: list[Any],
                   description: str = "") -> type[BaseModel]:
    ann = Literal[tuple(values)]
    return create_model(
        name,
        **{field_name: (ann, Field(description=description))},
    )


_B77_INFO = "https://huggingface.co/datasets/PolyAI/banking77/resolve/main/dataset_infos.json"
_B77_TEST = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"


def banking77(limit: int | None = None) -> EvalTask:
    """77-way customer-intent classification (PolyAI Banking77 test split).

    Loads the canonical CSV + label names directly — no trust_remote_code.
    """
    info = httpx.get(_B77_INFO, timeout=60, follow_redirects=True).json()
    names: list[str] = info["default"]["features"]["label"]["names"]
    rows = csv.reader(
        io.StringIO(httpx.get(_B77_TEST, timeout=60, follow_redirects=True).text),
        quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True,
    )
    next(rows)  # header
    examples = [
        EvalExample(state=text, labels={"intent": label})
        for text, label in rows
    ]
    model = _literal_model(
        "Banking77", "intent", names,
        "Classify the banking customer utterance into the correct intent.",
    )
    if limit:
        examples = examples[:limit]
    return EvalTask("banking77", model, examples,
                    description=f"77-way intent classification, {len(examples)} examples")


def _load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "This task needs the `datasets` package: pip install deciform[eval]"
        ) from e
    return load_dataset(*args, **kwargs)


def sst5(limit: int | None = None) -> EvalTask:
    """5-level sentiment rating -> score primitive (SetFit/sst5 test split)."""
    ds = _load_dataset("SetFit/sst5", split="test")
    model = create_model(
        "SST5",
        rating=(
            int,
            Field(ge=1, le=5,
                  description="Rate the sentiment of the movie review "
                              "from 1 (very negative) to 5 (very positive)."),
        ),
    )
    examples = [
        EvalExample(state=row["text"], labels={"rating": int(row["label"]) + 1})
        for row in ds
    ]
    if limit:
        examples = examples[:limit]
    return EvalTask("sst5", model, examples,
                    description=f"5-level sentiment rating, {len(examples)} examples")


def sst5_labeled(limit: int | None = None) -> EvalTask:
    """Same SST-5, but score levels carry semantic labels (very negative..very positive)."""
    task = sst5(limit)
    task.name = "sst5_labeled"
    task.model = create_model(
        "SST5Labeled",
        rating=(
            int,
            Field(ge=1, le=5,
                  description="Rate the sentiment of the movie review.",
                  json_schema_extra={
                      "level_labels": ["very negative", "negative", "neutral",
                                       "positive", "very positive"]
                  }),
        ),
    )
    return task


def rte(limit: int | None = None) -> EvalTask:
    """Recognizing Textual Entailment -> noul primitive (GLUE RTE validation)."""
    ds = _load_dataset("glue", "rte", split="validation")
    model = create_model(
        "RTE",
        entailment=(
            bool,
            Field(description="Does the premise entail the hypothesis?"),
        ),
    )
    examples = [
        EvalExample(
            state={"premise": row["sentence1"], "hypothesis": row["sentence2"]},
            labels={"entailment": row["label"] == 0},
        )
        for row in ds
    ]
    if limit:
        examples = examples[:limit]
    return EvalTask("rte", model, examples,
                    description=f"Binary entailment, {len(examples)} examples")


# ----------------------------------------------------------------------
# Fintech suite: procedurally generated expense states, deterministic labels
# ----------------------------------------------------------------------

_MERCHANTS = {
    "travel": ["Delta Airlines", "United Airlines", "Marriott Hotels", "Uber"],
    "meals": ["Sweetgreen", "Chipotle", "Starbucks", "Blue Bottle Coffee"],
    "software": ["GitHub", "Figma", "Datadog", "Linear"],
    "office_supplies": ["Staples", "Office Depot", "Uline"],
    "other": ["AMC Theatres", "Steam Games", "Liquor Store #5"],
}
_PERSONAL = ["AMC Theatres", "Steam Games", "Liquor Store #5"]


class FintechDecision(BaseModel):
    category: Literal["meals", "travel", "software", "office_supplies", "other"]
    policy_violation: bool = Field(
        description="Does this transaction violate the stated expense policy?"
    )
    severity: Literal["none", "minor", "major"] = Field(
        description="Severity of the policy violation, if any."
    )
    flags: list[Literal["duplicate", "over_limit", "missing_receipt", "personal"]]
    reviewer: Optional[Literal["auto_approve", "manager", "finance_team"]] = Field(
        default=None,
        description="Who must review this expense, if human review is needed.",
    )


def _gen_fintech_example(rng: random.Random) -> EvalExample:
    category = rng.choice(list(_MERCHANTS))
    merchant = rng.choice(_MERCHANTS[category])
    cap = {"travel": 500, "meals": 60, "software": 200, "office_supplies": 150,
           "other": 50}[category]
    amount = round(rng.uniform(10, cap * 2.2), 2)
    receipt = rng.random() < 0.75
    duplicate = rng.random() < 0.12
    personal = merchant in _PERSONAL

    over_limit = amount > cap
    missing_receipt = amount > 75 and not receipt
    violation = over_limit or missing_receipt or personal
    if not violation:
        severity = "none"
    elif over_limit and amount > 1.5 * cap or personal:
        severity = "major"
    else:
        severity = "minor"

    flags = [f for f, on in [
        ("duplicate", duplicate), ("over_limit", over_limit),
        ("missing_receipt", missing_receipt), ("personal", personal),
    ] if on]
    reviewer = (
        "finance_team" if severity == "major"
        else "manager" if severity == "minor"
        else None
    )

    state = {
        "merchant": merchant,
        "amount_usd": amount,
        "memo": rng.choice(
            ["client visit", "team offsite", "monthly subscription", "supplies", ""]
        ),
        "receipt_attached": receipt,
        "duplicate_transaction_id": duplicate,
        "expense_policy": {
            "category_cap_usd": cap,
            "requires_receipt_over_usd": 75,
            "personal_expenses_forbidden": True,
        },
    }
    return EvalExample(
        state=state,
        labels={
            "category": category,
            "policy_violation": violation,
            "severity": severity,
            "flags": flags,
            "reviewer": reviewer,
        },
    )


def fintech(n: int = 300, seed: int = 7) -> EvalTask:
    rng = random.Random(seed)
    examples = [_gen_fintech_example(rng) for _ in range(n)]
    return EvalTask("fintech", FintechDecision, examples,
                    description=f"Procedurally labeled expense decisions, {n} examples")


TASKS = {"banking77": banking77, "sst5": sst5, "sst5_labeled": sst5_labeled,
         "rte": rte, "fintech": fintech}
