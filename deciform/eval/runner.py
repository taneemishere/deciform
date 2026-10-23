"""Eval runner: drive a task's labeled examples through Deciform.extract
on a given backend, in parallel, with retries — then score every field.

Produces an EvalReport: per-field accuracy/ECE/Brier/coverage curves, latency,
error counts, plus the raw FieldRecords (which feed EscalationPolicy.fit).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from deciform.backends.base import DecisionBackend, GenerativeBackend
from deciform.compiler import compile
from deciform.core import Deciform
from deciform.eval.metrics import (
    FieldRecord, accuracy, auarc, brier, coverage_curve, ece, latency_stats,
)
from deciform.policy import EscalationPolicy
from deciform.questions import ChoiceQuestion
from deciform.results import Answer


@dataclass
class EvalReport:
    task: str
    backend: str
    n_examples: int
    n_errors: int
    field_metrics: dict[str, dict[str, Any]]
    example_latency: dict[str, float]
    wall_seconds: float
    records: list[FieldRecord] = field(default_factory=list)
    error_kinds: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "backend": self.backend,
            "n_examples": self.n_examples,
            "n_errors": self.n_errors,
            "field_metrics": self.field_metrics,
            "example_latency": self.example_latency,
            "wall_seconds": self.wall_seconds,
            "error_kinds": self.error_kinds,
            "records": [asdict(r) for r in self.records],
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def calibration_rows(self) -> dict[str, list[tuple[float, bool]]]:
        """field path -> [(top_probability, correct)] for EscalationPolicy.fit."""
        rows: dict[str, list[tuple[float, bool]]] = {}
        for r in self.records:
            if not r.skipped:
                rows.setdefault(r.path, []).append((r.top_probability, r.correct))
        return rows


def _get(tree: Any, path: str) -> Any:
    for part in path.split("."):
        if tree is None:
            return None
        tree = tree.get(part) if isinstance(tree, dict) else getattr(tree, part, None)
    return tree


def _field_probs(a: Answer | None, bounds=None) -> dict | None:
    if a is None:
        return None
    if a.type == "noul":
        return {"yes": a.value, "no": 1.0 - a.value}
    if a.type == "score" and a.probabilities and bounds:
        lo, _ = bounds
        return {str(lo + int(k)): p for k, p in a.probabilities.items()}
    return a.probabilities


def _extract_once(sf: Deciform, model, state: Any, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            return sf.extract(model, state)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else 0
            if attempt == max_retries - 1 or status not in (429, 500, 502, 503, 504):
                raise
            ra = e.response.headers.get("retry-after")
            try:
                wait = float(ra) if ra else (20.0 if status == 429 else 2.0 * (2 ** attempt))
            except ValueError:
                wait = 20.0
            time.sleep(min(wait, 90.0))
        except httpx.TransportError:
            if attempt == max_retries - 1:
                raise
            time.sleep(2.0 * (2 ** attempt))
    raise RuntimeError("unreachable")


def run_task(
    task,
    backend: DecisionBackend,
    *,
    escalation: EscalationPolicy | None = None,
    fallback: GenerativeBackend | None = None,
    workers: int = 8,
    max_retries: int = 3,
    check_order_flip: bool = False,
    gate_threshold: float = 0.5,
) -> EvalReport:
    sf = Deciform(backend=backend, fallback=fallback, escalation=escalation,
                  gate_threshold=gate_threshold)
    program = compile(task.model)
    plans = {p.path: p for p in program.fields}
    choice_qs = {
        p.path: program.questions[p.question_names[0]]
        for p in program.fields
        if p.kind == "choice" and len(p.question_names) == 1
    }

    records: list[FieldRecord] = []
    latencies: list[float] = []
    flips: dict[str, list[bool]] = {}
    error_kinds: dict[str, int] = {}
    t0 = time.perf_counter()

    def work(ex):
        res = _extract_once(sf, task.model, ex.state, max_retries)
        per_field = {}
        for tr in res.fields:
            if tr.path not in ex.labels:
                continue
            label = ex.labels[tr.path]
            val = _get(res.value.model_dump(), tr.path)
            if isinstance(label, list) and isinstance(val, list):
                correct = set(map(str, val)) == set(map(str, label))
            else:
                correct = val == label
            plan = plans.get(tr.path)
            per_field[tr.path] = FieldRecord(
                path=tr.path, value=val, label=label,
                probabilities=_field_probs(
                    tr.answer, plan.bounds if plan else None),
                top_probability=tr.answer.top_probability if tr.answer else 0.0,
                correct=correct, skipped=tr.skipped, escalated=tr.escalated,
            )

        # order-flip probe, paired per example: re-ask each single-question
        # choice field with the criteria reversed
        flip_results = {}
        if check_order_flip:
            for path, q in choice_qs.items():
                orig = per_field.get(path)
                if orig is None or orig.skipped:
                    continue
                rev = ChoiceQuestion(q.name, q.instructions,
                                     dict(reversed(list(q.criteria.items()))))
                a = backend.evaluate(ex.state, [rev])[q.name]
                flip_results[path] = (orig.value != a.value)
        return per_field, res.total_latency_ms, flip_results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, ex): ex for ex in task.examples}
        for fut in as_completed(futures):
            ex = futures[fut]
            try:
                per_field, ms, flip_results = fut.result()
                records.extend(per_field.values())
                latencies.append(ms)
                for path, flipped in flip_results.items():
                    flips.setdefault(path, []).append(flipped)
            except Exception as e:
                kind = type(e).__name__
                error_kinds[kind] = error_kinds.get(kind, 0) + 1
                # A crashed example still counts, else accuracy measures
                # survivors only and a flaky endpoint inflates every metric.
                for path, label in ex.labels.items():
                    records.append(FieldRecord(
                        path=path, value=None, label=label, probabilities=None,
                        top_probability=0.0, correct=False, skipped=True))

    field_metrics: dict[str, dict[str, Any]] = {}
    paths = sorted({r.path for r in records})
    for path in paths:
        recs = [r for r in records if r.path == path]
        m = {
            "accuracy": accuracy(recs),
            "ece": ece(recs),
            "brier": brier(recs),
            "auarc": auarc(recs),
            "coverage_curve": coverage_curve(recs),
            "n": len(recs),
        }
        if path in flips:
            m["order_flip_rate"] = sum(flips[path]) / len(flips[path])
        plan = plans.get(path)
        if plan and plan.gated_by:
            # Presence-gate quality, measured apart from the child answer:
            # predicted present = the gate fired (field not skipped),
            # actual present = the label exists.
            tp = sum(1 for r in recs if not r.skipped and r.label is not None)
            fp = sum(1 for r in recs if not r.skipped and r.label is None)
            fn = sum(1 for r in recs if r.skipped and r.label is not None)
            m["gate_precision"] = tp / (tp + fp) if tp + fp else None
            m["gate_recall"] = tp / (tp + fn) if tp + fn else None
            m["gate_pred_present"] = tp + fp
            m["gate_actual_present"] = tp + fn
        field_metrics[path] = m

    return EvalReport(
        task=task.name, backend=backend.name,
        n_examples=len(task.examples), n_errors=sum(error_kinds.values()),
        field_metrics=field_metrics,
        example_latency=latency_stats(latencies),
        wall_seconds=time.perf_counter() - t0,
        records=records, error_kinds=error_kinds,
    )
