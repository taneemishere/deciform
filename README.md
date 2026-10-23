# DeciForm: Structured Field Extraction Without Generative Decoding

**Structured objects from System One decision models. Schema-guaranteed, calibrated, and field-level confident.**

`jsonformer` (2023) proved you don't trust a language model with structural tokens: fill the JSON skeleton in code, generate only content tokens. `deciform` applies the same philosophy to the System One era: don't use a *generative* model for typed fields at all. Compile your schema into a batched program of `choice` / `noul` / `score` questions, run it on any decision-model backend, and escalate only the fields the model can't confidently decide to an LLM.

## When to reach for it

Your pipeline makes thousands of small structured decisions a day. Triage a ticket, flag a transaction, route a document, grade a lead. Each one is currently either an LLM call you overpay for and can't calibrate, or a decision-model call plus a few hundred lines of glue you wrote by hand, mapping fields to questions, batching, gating optional branches, thresholding confidence, reassembling the object, logging what happened. `deciform` is that layer, written once. If your whole output is a single yes/no, use the raw API; if your schema is mostly free text, you want a structured-output tool, not this.

## Example

```python
from pydantic import BaseModel, Field
from typing import Literal
from deciform import Deciform, EscalationPolicy, SystemOneBackend
from deciform.backends import OpenAIGenerativeBackend

class TicketTriage(BaseModel):
    category: Literal["billing", "bug", "account", "feature_request", "other"]
    is_urgent: bool
    sentiment: Literal["angry", "neutral", "positive"]
    priority: int = Field(ge=1, le=5)                    # bounded int -> score
    needs_human: Literal["none", "support", "engineering"] | None  # Optional -> presence gate
    reply_draft: str = ""                                # free text -> fallback LLM

sf = Deciform(
    backend=SystemOneBackend(model=SystemOneBackend.ELEPHANT),  # any /v1/systemone endpoint
    fallback=OpenAIGenerativeBackend(                           # any OpenAI-compatible LLM
        base_url="https://your-llm-endpoint/v1",
        api_key="...", model="your-chat-model"),
    escalation=EscalationPolicy(threshold=0.8, debias_k=3),
)

result = sf.extract(TicketTriage, state=ticket_json)
```

What you get back is a validated object *plus* a per-field receipt:

```python
result.value      # TicketTriage(category='billing', is_urgent=True, ...)
result.fields[2]  # FieldTrace for 'sentiment': backend='elephant', answer.top_probability=0.94
result.escalated  # ['priority', 'reply_draft']   <- fell below 0.8 / not a decision primitive
result.usage      # {'requests': 3, 'prompt_tokens': ..., 'total_tokens': ...}
```

Runnable version with zero API keys in `examples/ticket_triage.py`; a real-endpoint wiring in `examples/expense_review.py`.

## How it works

```
sf.extract(Model, state)
        │   schema walk              ┌───────────────────┐
   compile(Model)  ────────────────► │ question program   │  rounds + gates
        │                            └─────────┬─────────┘
        │                              batched evaluate()   one request / round
        │                            ┌─────────▼─────────┐
        │                            │ probabilities in   │  per-field answers
        │   confident fields         └──────┬──────┬──────┘  uncertain fields
        └────── keep decision               │      └─► GenerativeBackend (escalation)
        └─────────────────────►  assemble → Pydantic validate → DecisionResult
```

| Schema element | Compiled to |
|---|---|
| `Literal[...]` / `Enum` | `choice` (option-order debiasing optional) |
| `bool` | `noul` |
| bounded `int` (`ge`/`le`/`Field`) | `score` (ordinal levels; `level_labels` via `json_schema_extra`) |
| `str` / unbounded numbers | generative fallback, schema-hinted |
| `list[Literal[...]]` | `multinoul`, one `noul` per member |
| `Optional[T]` | presence `noul` → child questions in a lazy later round |
| nested `BaseModel` | question group sharing the state prefix |

## The ops layer the ecosystem leaves to you

- **Calibrated escalation**: `EscalationPolicy(threshold=…, per_field={…})`; `policy.fit(calibration_rows, target_error=0.05)` picks per-field τ on *your* labeled data (Wilson-bound acceptance, conformal-flavored, honest about what the model can't promise).
- **Order debiasing**: `debias_k=3` rotates options and averages probabilities (the literature reports ~30% flip rates on some open models; measured 2–9% on elephant/mammoth).
- **Per-field traces**: every field's questions, backend, top probability, latency, escalated-or-not, skipped-by-gate-or-not.
- **Token accounting**: `result.usage` aggregates prompt/completion/total tokens across every call in the extract (when the API reports it).
- **Async**: `await sf.aextract(Model, state)` runs the sync pipeline in a worker thread; rounds are sequential by design, so the win is concurrency across calls in async services.
- **Backend-agnostic**: `SystemOneBackend` for any `/v1/systemone` provider; `OpenAIGenerativeBackend` for any OpenAI-compatible chat endpoint; `LLMAsJevBackend` turns any chat LLM with `logprobs` into a decision backend via first-token option-letter readout.

## Measured (live, full test splits)

| task | elephant-1 (9B) | mammoth-1 (27B) |
|---|---|---|
| banking77 intent (77 options) | acc .929 · ece .013 · flip .024 | acc .937 · ece .010 · flip .020 |
| RTE entailment | .895 | .946 |
| fintech suite (5 fields) | violation .890 · flags .781 | violation .945 · flags .915 |

Plus: LLM-as-Jev readout verified on a 30B chat model; escalation Pareto showing a +1.5pp bump at ~25% fallback spend when the fallback is strong enough to deserve it. Full numbers and analysis in `REPORT.md`; raw records in `results/`.

## Install

```bash
pip install -e .          # core: pydantic + httpx only
pip install -e .[eval]    # benchmark harness: datasets + matplotlib
pip install -e .[local]   # future local-encoder backends: torch + transformers
```

## Eval harness

```bash
# benchmarks: banking77 · sst5(_labeled) · rte · procedural fintech
python scripts/run_eval.py --tasks fintech rte --backends elephant mammoth \
    --limit 500 --workers 8 --out results --fit-error 0.05

# escalation Pareto: decision model decides, LLM covers the low-confidence tail
python scripts/escalation_experiment.py --task fintech \
    --llm-model nvidia/nemotron-3-ultra-550b-a55b --tag ultra

# summaries + coverage plots
python scripts/summarize.py --results results
```

## Status

v0.x. Core pipeline, both backend kinds, eval suite, and threshold fitting are live-verified. Still coming: a published PyPI name and richer cost metering per field.

## Lineage

Named in homage to [jsonformer](https://github.com/1rgs/jsonformer) by Rahul Sengottuvelu. Same thesis, next model class: *the structure is yours; let the model do only what it must.*
