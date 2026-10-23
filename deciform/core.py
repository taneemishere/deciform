"""Deciform: compile a schema into typed decisions, assemble a validated object.

Pipeline:

    extract(Model, state)
      1. compile(Model) -> question program (rounds via Optional/dependency gates)
      2. evaluate each round on the decision backend, batching independent
         questions into one call (shared state prefix)
      3. policy check per field: confident -> keep; unconfident -> generative
         fallback (or flag low_confidence when no fallback is attached)
      4. assemble the object tree, Pydantic-validate, attach per-field traces
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, TypeAdapter

from deciform.backends.base import DecisionBackend, GenerativeBackend
from deciform.compiler import FieldPlan, QuestionProgram, compile
from deciform.policy import EscalationPolicy
from deciform.questions import ChoiceQuestion, Question
from deciform.results import Answer, DecisionResult, FieldTrace


def _set_path(tree: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    node = tree
    for parts_item in parts[:-1]:
        node = node.setdefault(parts_item, {})
    node[parts[-1]] = value


def _leaf_schema(plan: FieldPlan) -> dict[str, Any]:
    """JSON-schema fragment for the field's annotation (enum/type/bounds)."""
    try:
        return TypeAdapter(plan.annotation).json_schema()
    except Exception:
        return {}


def _sum_usage(usages: list[dict]) -> dict[str, int]:
    """Merge token accounting across all backend calls (key styles vary).
    Callers append one entry per request ({} when unreported), so
    ``requests`` counts real calls, not just calls that reported usage."""
    out: dict[str, int] = {"requests": len(usages)}
    for u in usages:
        for src, dst in (
            ("prompt_tokens", "prompt_tokens"), ("input_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"), ("output_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            v = u.get(src)
            if isinstance(v, (int, float)):
                out[dst] = out.get(dst, 0) + int(v)
    return out


class Deciform:
    """Extract schema-validated objects using a decision-model backend.

    backend: any DecisionBackend (SystemOneBackend, local encoders,
             LLMAsJevBackend readout, or your own Protocol impl)
    fallback: optional GenerativeBackend for free-text fields and escalations
    escalation: EscalationPolicy controlling per-field confidence thresholds
                and choice-order debiasing
    gate_threshold: presence probability at which an Optional field's gate
                    fires. Lower it (e.g. 0.3) when gates under-fire; the
                    cost is extra round trips for branches that stay empty.
    """

    def __init__(
        self,
        backend: DecisionBackend,
        fallback: GenerativeBackend | None = None,
        escalation: EscalationPolicy | None = None,
        gate_threshold: float = 0.5,
    ):
        self.backend = backend
        self.fallback = fallback
        self.escalation = escalation or EscalationPolicy()
        self.gate_threshold = gate_threshold
        self._programs: dict[type, QuestionProgram] = {}  # model -> compiled program

    # ------------------------------------------------------------------
    # Evaluation rounds
    # ------------------------------------------------------------------

    def _evaluate(
        self, state: Any, questions: list[Question]
    ) -> tuple[dict[str, Answer], float, dict[str, float]]:
        """Evaluate a batch, with choice-order debiasing.

        Returns (answers, latency_ms, per_question_ms) where the last maps
        each question to its share of the call's latency (rotations cost more).
        """
        k = max(1, self.escalation.debias_k)
        if k == 1 or not any(isinstance(q, ChoiceQuestion) for q in questions):
            start = time.perf_counter()
            answers = self.backend.evaluate(state, questions)
            ms = (time.perf_counter() - start) * 1000
            share = ms / max(1, len(questions))
            return answers, ms, {q.name: share for q in questions}

        # Expand each choice question into k cyclic option rotations; the
        # merged answer averages probabilities across orders.
        expanded: list[Question] = []
        weights: dict[str, int] = {}  # original name -> variants sent
        for q in questions:
            if not isinstance(q, ChoiceQuestion):
                expanded.append(q)
                weights[q.name] = 1
                continue
            keys = list(q.criteria)
            n_rot = min(k, len(keys))  # past that, rotations repeat and double-weight
            for i in range(n_rot):
                rotated = keys[i:] + keys[:i]
                expanded.append(
                    ChoiceQuestion(name=f"{q.name}#rot{i}", instructions=q.instructions,
                                   criteria={kk: q.criteria[kk] for kk in rotated})
                )
            weights[q.name] = n_rot

        start = time.perf_counter()
        raw = self.backend.evaluate(state, expanded)
        latency_ms = (time.perf_counter() - start) * 1000

        answers: dict[str, Answer] = {}
        for q in questions:
            if not isinstance(q, ChoiceQuestion):
                answers[q.name] = raw[q.name]
                continue
            n_rot = weights[q.name]
            merged: dict[str, float] = {}
            for i in range(n_rot):
                for opt, p in (raw[f"{q.name}#rot{i}"].probabilities or {}).items():
                    merged[opt] = merged.get(opt, 0.0) + p / n_rot
            best = max(merged, key=merged.get)
            answers[q.name] = Answer(
                type="choice", value=best, probabilities=merged,
                confidence=merged[best], raw=raw[f"{q.name}#rot0"].raw,
            )
        total = sum(weights.values())
        return answers, latency_ms, {n: latency_ms * w / total for n, w in weights.items()}

    def _run_rounds(self, state: Any, program: QuestionProgram) -> dict[str, Any]:
        """Fire question rounds, respecting dependency gates.

        Returns (answers, skipped_names, latency_ms, usages, per_question_ms).
        """
        answers: dict[str, Answer] = {}
        latency_ms = 0.0
        usages: list[dict] = []
        q_ms: dict[str, float] = {}
        pending = set(program.questions)
        skipped: set[str] = set()
        gate_answers: dict[str, float] = {}

        while pending:
            ready, blocked = [], []
            for name in program.questions:  # insertion order -> deterministic batches
                if name not in pending:
                    continue
                gate = program.depends_on.get(name)
                if gate is None or (gate in gate_answers
                                    and gate_answers[gate] >= self.gate_threshold):
                    ready.append(name)
                elif gate in skipped or (gate in gate_answers
                                         and gate_answers[gate] < self.gate_threshold):
                    blocked.append(name)  # gate fired false / gated-off -> never ask
                    skipped.add(name)
                # else: gate unanswered this pass (nested optionals)

            if ready:
                batch = [program.questions[n] for n in ready]
                round_answers, ms, round_q_ms = self._evaluate(state, batch)
                latency_ms += ms
                q_ms.update(round_q_ms)
                usages.append(getattr(self.backend, "last_usage", None) or {})
                for n in ready:
                    answers[n] = round_answers[n]
                    pending.discard(n)
                    if n.endswith("__present"):
                        gate_answers[n] = round_answers[n].value
            for n in blocked:
                pending.discard(n)
            if not ready and not blocked:
                break  # unreachable in practice; safety against cycles

        return answers, skipped, latency_ms, usages, q_ms

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _convert(self, plan: FieldPlan, answers: dict[str, Answer]) -> tuple[Any, Answer | None]:
        """Turn a field's answers into a python value."""
        if plan.kind == "choice":
            a = answers[plan.question_names[0]]
            return plan.value_map.get(a.value, a.value), a
        if plan.kind == "noul":
            a = answers[plan.question_names[0]]
            return a.value >= 0.5, a
        if plan.kind == "score":
            a = answers[plan.question_names[0]]
            lo, hi = plan.bounds
            # a.value is the expected LEVEL INDEX (0..k-1); map back to value space
            return int(round(min(max(lo + a.value, lo), hi))), a
        if plan.kind == "multinoul":
            chosen, probs = [], {}
            for qname in plan.question_names:
                member = qname.split("::", 1)[1]
                a = answers[qname]
                probs[member] = a.value
                if a.value >= 0.5:
                    chosen.append(plan.value_map.get(member, member))
            rep = Answer(type="multinoul", value=chosen, probabilities=probs)
            return chosen, rep
        return None, None  # generative handled by caller

    def extract(self, model: type[BaseModel], state: Any) -> DecisionResult:
        program = self._programs.get(model)
        if program is None:
            program = compile(model)
            self._programs[model] = program

        if any(f.kind == "generative" for f in program.fields) and self.fallback is None:
            generative_paths = [f.path for f in program.fields if f.kind == "generative"]
            raise ValueError(
                f"Fields {generative_paths} have no decision primitive "
                "(free text, unbounded number, or unsupported type). "
                "Attach a GenerativeBackend as `fallback` or remove them."
            )

        answers, skipped, latency_ms, usages, q_ms = self._run_rounds(state, program)

        tree: dict = {}
        traces: list[FieldTrace] = []
        escalated: list[str] = []

        for plan in program.fields:
            gate = plan.gated_by
            gate_fired_false = (
                gate is not None and gate in answers
                and answers[gate].value < self.gate_threshold
            )
            gate_never_ran = gate is not None and gate not in answers
            if gate_fired_false or gate_never_ran or any(q in skipped for q in plan.question_names):
                # Whole optional branch absent -> set its root to the declared
                # default, not a dict of Nones per leaf. When the direct gate
                # never ran, an ancestor gate already wrote the subtree
                # default -> write nothing (writing would clobber it anyway).
                if gate_fired_false and gate in program.gate_roots:
                    root, default = program.gate_roots[gate]
                    _set_path(tree, root, default)
                elif gate is None:
                    _set_path(tree, plan.path, plan.default)
                # carry the answered gate (or its deciding ancestor) so
                # "absent" keeps a real confidence in the trace
                g, gate_answer = gate, None
                while g is not None and gate_answer is None:
                    gate_answer = answers.get(g)
                    g = program.depends_on.get(g)
                traces.append(FieldTrace(path=plan.path, questions=plan.question_names,
                                         backend=self.backend.name, answer=gate_answer,
                                         skipped=True))
                continue

            share_ms = sum(q_ms.get(q, 0.0) for q in plan.question_names)
            start = time.perf_counter()
            if plan.kind == "generative":
                value = self.fallback.produce(
                    state, instructions=f"Provide the value for '{plan.path}'.",
                    context=tree, schema_hint=_leaf_schema(plan),
                )
                usages.append(getattr(self.fallback, "last_usage", None) or {})
                _set_path(tree, plan.path, value)
                traces.append(FieldTrace(
                    path=plan.path, questions=[], backend=f"generative:{self.fallback.name}",
                    answer=None, escalated=True,
                    latency_ms=(time.perf_counter() - start) * 1000,
                ))
                continue

            value, rep = self._convert(plan, answers)
            if rep and self.escalation.should_escalate(plan.path, rep):
                if self.fallback is not None:
                    value = self.fallback.produce(
                        state,
                        instructions=f"Provide the value for '{plan.path}'.",
                        context=tree, schema_hint=_leaf_schema(plan),
                    )
                    usages.append(getattr(self.fallback, "last_usage", None) or {})
                    _set_path(tree, plan.path, value)
                    escalated.append(plan.path)
                    traces.append(FieldTrace(
                        path=plan.path, questions=plan.question_names,
                        backend=f"generative:{self.fallback.name}", answer=rep,
                        escalated=True,
                        latency_ms=share_ms + (time.perf_counter() - start) * 1000,
                    ))
                    continue
                traces.append(FieldTrace(
                    path=plan.path, questions=plan.question_names,
                    backend=self.backend.name, answer=rep, low_confidence=True,
                    latency_ms=share_ms,
                ))
            else:
                traces.append(FieldTrace(
                    path=plan.path, questions=plan.question_names,
                    backend=self.backend.name, answer=rep,
                    latency_ms=share_ms,
                ))
            _set_path(tree, plan.path, value)

        value = model.model_validate(tree)

        return DecisionResult(
            value=value, fields=traces, escalated=escalated,
            total_latency_ms=latency_ms + sum(t.latency_ms for t in traces),
            usage=_sum_usage(usages),
        )

    async def aextract(self, model: type[BaseModel], state: Any) -> DecisionResult:
        """Async wrapper — runs extract() in a worker thread.

        Rounds are sequential by design (gated dependencies), so the win is
        concurrency across calls in async services, not within one extract.
        """
        import asyncio
        return await asyncio.to_thread(self.extract, model, state)
