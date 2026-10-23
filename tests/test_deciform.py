"""End-to-end tests with a fake decision backend — no API key required."""

from __future__ import annotations

from typing import Any, Literal, Optional

import pytest
from pydantic import BaseModel, Field

from deciform import EscalationPolicy, Deciform
from deciform.questions import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion
from deciform.results import Answer


class FakeBackend:
    """DecisionBackend with scripted per-question probabilities."""

    name = "fake"

    def __init__(self, script: dict[str, Answer]):
        self.script = script
        self.calls: list[list[str]] = []  # question names per evaluate() call

    def evaluate(self, state: Any, questions: list[Question]) -> dict[str, Answer]:
        self.calls.append([q.name for q in questions])
        out = {}
        for q in questions:
            base = q.name.split("#rot")[0]  # debias variants map to base script
            if base in self.script:
                a = self.script[base]
                out[q.name] = Answer(type=a.type, value=a.value,
                                     probabilities=a.probabilities,
                                     confidence=a.confidence)
            elif isinstance(q, ChoiceQuestion):
                first = next(iter(q.criteria))
                probs = {k: (0.9 if k == first else 0.1 / (len(q.criteria) - 1))
                         for k in q.criteria}
                out[q.name] = Answer(type="choice", value=first, probabilities=probs)
            elif isinstance(q, NoulQuestion):
                out[q.name] = Answer(type="noul", value=0.9)
            elif isinstance(q, ScoreQuestion):
                out[q.name] = Answer(type="score", value=1.0,
                                     probabilities={str(i): 1.0 / len(q.criteria)
                                                    for i in range(len(q.criteria))})
        return out


class FakeFallback:
    name = "fakegen"

    def __init__(self, value: Any = "GENERATED"):
        self.value = value
        self.prompts: list[str] = []

    def produce(self, state: Any, instructions: str, context: dict,
                schema_hint: dict | None = None) -> Any:
        self.prompts.append(instructions)
        return self.value


class Ticket(BaseModel):
    team: Literal["billing", "technical_support"]
    refund_requested: bool
    urgency: int = Field(ge=1, le=3)
    tags: list[Literal["vip", "angry", "duplicate"]]
    summary: str = ""


class Profile(BaseModel):
    name_literal: Literal["alice", "bob"]
    address: Optional["Address"] = None


class Address(BaseModel):
    city_literal: Literal["nyc", "la"]
    zip_known: bool


def test_compile_maps_field_types():
    from deciform import compile as sfcompile
    from pydantic import BaseModel as BM, Field as F

    class M(BM):
        cat: Literal["a", "b"]
        flag: bool
        level: int = F(ge=1, le=5)
        picks: list[Literal["x", "y"]]
        note: str = "hi"

    prog = sfcompile(M)
    kinds = {f.path: f.kind for f in prog.fields}
    assert kinds == {
        "cat": "choice", "flag": "noul", "level": "score",
        "picks": "multinoul", "note": "generative",
    }
    assert isinstance(prog.questions["cat"], ChoiceQuestion)
    assert isinstance(prog.questions["flag"], NoulQuestion)
    assert isinstance(prog.questions["level"], ScoreQuestion)
    assert isinstance(prog.questions["picks::x"], NoulQuestion)
    assert "note" not in prog.questions  # generative fields ask nothing


def test_extract_assembles_validated_object():
    backend = FakeBackend({
        "team": Answer(type="choice", value="billing",
                       probabilities={"billing": 0.95, "technical_support": 0.05}),
        "refund_requested": Answer(type="noul", value=0.88),
        "urgency": Answer(type="score", value=1.0,  # index space: level 1 -> value 2
                          probabilities={"0": 0.1, "1": 0.8, "2": 0.1}),
        "tags::vip": Answer(type="noul", value=0.9),
        "tags::angry": Answer(type="noul", value=0.2),
        "tags::duplicate": Answer(type="noul", value=0.7),
    })
    fb = FakeFallback("charged twice, wants refund")
    sf = Deciform(backend=backend, fallback=fb)
    res = sf.extract(Ticket, state="double charge please refund")

    v = res.value
    assert v.team == "billing"
    assert v.refund_requested is True
    assert v.urgency == 2  # rounded weighted score
    assert v.tags == ["vip", "duplicate"]
    assert v.summary == "charged twice, wants refund"
    # one batched decision call, all independent questions together
    assert len(backend.calls) == 1
    assert "summary" not in backend.calls[0]
    assert res.fields[-1].backend == "generative:fakegen"


def test_generative_field_requires_fallback():
    backend = FakeBackend({})
    sf = Deciform(backend=backend)
    with pytest.raises(ValueError, match="GenerativeBackend"):
        sf.extract(Ticket, state="x")


def test_optional_gate_fires_second_round():
    backend = FakeBackend({
        "address__present": Answer(type="noul", value=0.92),
        "address.city_literal": Answer(
            type="choice", value="nyc",
            probabilities={"nyc": 0.8, "la": 0.2}),
        "address.zip_known": Answer(type="noul", value=0.6),
        "name_literal": Answer(type="choice", value="alice",
                               probabilities={"alice": 0.9, "bob": 0.1}),
    })
    sf = Deciform(backend=backend)
    res = sf.extract(Profile, state="alice from nyc")

    # round 1 asks independent + presence; round 2 asks gated children
    assert set(backend.calls[0]) == {"name_literal", "address__present"}
    assert set(backend.calls[1]) == {"address.city_literal", "address.zip_known"}
    assert res.value.address.city_literal == "nyc"
    assert res.value.address.zip_known is True


def test_optional_gate_false_skips_children():
    backend = FakeBackend({
        "address__present": Answer(type="noul", value=0.1),
        "name_literal": Answer(type="choice", value="bob",
                               probabilities={"alice": 0.1, "bob": 0.9}),
    })
    sf = Deciform(backend=backend)
    res = sf.extract(Profile, state="bob, no address")

    assert len(backend.calls) == 1  # children never asked
    assert res.value.address is None
    child_traces = [t for t in res.fields if t.path.startswith("address.")]
    assert all(t.skipped for t in child_traces)


def test_gate_threshold_lowered_fires_children():
    # P(present)=0.35: blocked at the default 0.5 gate, fires at 0.3.
    backend = FakeBackend({
        "address__present": Answer(type="noul", value=0.35),
        "name_literal": Answer(type="choice", value="bob",
                               probabilities={"alice": 0.1, "bob": 0.9}),
        "address.city_literal": Answer(
            type="choice", value="nyc", probabilities={"nyc": 0.8, "la": 0.2}),
        "address.zip_known": Answer(type="noul", value=0.7),
    })
    sf = Deciform(backend=backend, gate_threshold=0.3)
    res = sf.extract(Profile, state="bob in nyc")

    assert len(backend.calls) == 2  # round 2 ran
    assert res.value.address is not None
    assert res.value.address.city_literal == "nyc"


class NestInner(BaseModel):
    flag: bool
    note: str = "x"  # generative leaf inside a gated branch


class NestOuter(BaseModel):
    a: bool
    inner: Optional[NestInner] = None


class NestM(BaseModel):
    name: Literal["a", "b"]
    outer: Optional[NestOuter] = None


def test_nested_optional_outer_absent():
    backend = FakeBackend({
        "outer__present": Answer(type="noul", value=0.1),  # outer absent
        "name": Answer(type="choice", value="a", probabilities={"a": 0.9, "b": 0.1}),
    })
    fb = FakeFallback()
    res = Deciform(backend=backend, fallback=fb).extract(NestM, state="x")

    assert res.value.outer is None
    assert len(backend.calls) == 1  # only round 1 ran
    assert fb.prompts == []  # no produce() for fields under a dead gate
    assert all(t.skipped for t in res.fields if t.path.startswith("outer"))
    # the trace carries the gate that decided the absence
    assert all(t.answer is not None and t.answer.value == 0.1
               for t in res.fields if t.path.startswith("outer"))


def test_nested_optional_inner_absent():
    backend = FakeBackend({
        "outer__present": Answer(type="noul", value=0.9),
        "outer.inner__present": Answer(type="noul", value=0.1),
        "outer.a": Answer(type="noul", value=0.8),
        "name": Answer(type="choice", value="a", probabilities={"a": 0.9, "b": 0.1}),
    })
    res = Deciform(backend=backend, fallback=FakeFallback()).extract(NestM, state="x")

    assert res.value.outer.a is True
    assert res.value.outer.inner is None
    assert len(backend.calls) == 2  # r1: name+outer gate; r2: a+inner gate; inner's children skipped


def test_eval_counts_crashed_examples_as_wrong():
    from deciform.eval.runner import run_task
    from deciform.eval.tasks import EvalExample, EvalTask

    class Failing:
        name = "fails"

        def evaluate(self, state, questions):
            raise RuntimeError("boom")

    class M(BaseModel):
        team: Literal["a", "b"]

    task = EvalTask(name="t", model=M,
                    examples=[EvalExample("x", {"team": "a"}), EvalExample("y", {"team": "b"})])
    rep = run_task(task, Failing(), workers=1, max_retries=1)

    assert rep.n_errors == 2
    m = rep.field_metrics["team"]
    assert m["n"] == 2 and m["accuracy"] == 0.0  # not survivors-only


def test_gate_question_asks_applicability():
    from deciform import compile as sfcompile

    prog = sfcompile(Profile)
    q = prog.questions["address__present"]
    assert "assigned" in q.instructions  # applicability, not literal presence
    assert "specify" not in q.instructions


def test_escalation_threshold_sends_to_fallback():
    fb = FakeFallback("technical_support")
    backend = FakeBackend({
        "team": Answer(type="choice", value="billing",
                       probabilities={"billing": 0.55, "technical_support": 0.45}),
        "refund_requested": Answer(type="noul", value=0.95),
        "urgency": Answer(type="score", value=0.0,  # index 0 -> value lo=1
                          probabilities={"0": 0.9, "1": 0.05, "2": 0.05}),
        "tags::vip": Answer(type="noul", value=0.99),
        "tags::angry": Answer(type="noul", value=0.01),
        "tags::duplicate": Answer(type="noul", value=0.01),
        "summary": Answer(type="noul", value=0.0),
    })
    sf = Deciform(backend=backend, fallback=fb,
                      escalation=EscalationPolicy(threshold=0.8))
    res = sf.extract(Ticket, state="x")

    assert res.value.team == "technical_support"  # fallback value wins
    assert res.escalated == ["team"]  # only 0.55 < 0.8 escalates
    # urgency (max prob 0.9) and confident-negative nouls (0.99) are kept
    assert res.value.urgency == 1
    assert res.value.tags == ["vip"]


def test_escalation_without_fallback_marks_low_confidence():
    backend = FakeBackend({
        "team": Answer(type="choice", value="billing",
                       probabilities={"billing": 0.55, "technical_support": 0.45}),
    })

    class M(BaseModel):
        team: Literal["billing", "technical_support"]

    sf = Deciform(backend=backend, escalation=EscalationPolicy(threshold=0.8))
    res = sf.extract(M, state="x")
    assert res.value.team == "billing"  # kept, but flagged
    assert res.fields[0].low_confidence is True


def test_policy_fit_thresholds():
    policy = EscalationPolicy()
    calib = {
        # all correct -> smallest tau = the lowest observed prob
        "clean": [(0.6, True)] * 12,
        # low-confidence band is wrong -> tau must land above it
        "noisy_low": [(0.55, False)] * 5 + [(0.92, True)] * 12,
        # never clean -> no tau qualifies -> always escalate
        "always_wrong": [(0.7, False)] * 15,
    }
    fitted = policy.fit(calib, target_error=0.10)

    assert fitted["clean"] == 0.6
    assert fitted["noisy_low"] == 0.92
    assert fitted["always_wrong"] > 1.0
    assert policy.should_escalate("always_wrong", Answer(type="noul", value=0.99))
    assert not policy.should_escalate(
        "clean", Answer(type="choice", value="x", probabilities={"x": 0.7}))


def _lp(tokens):
    """Canned logprobs `choice` payload: [(token, prob), ...] at position 0."""
    import math
    return {"logprobs": {"content": [{
        "token": tokens[0][0], "logprob": math.log(tokens[0][1]),
        "top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in tokens],
    }]}}


def test_llm_readout_letter_mass():
    from deciform.backends.llm import LLMAsJevBackend

    backend = LLMAsJevBackend(base_url="http://x", api_key=None, model="m")

    # noul: YES .8 / NO .2 -> normalized P(true) = .8
    a = backend._answer_from_logprobs(
        _lp([("YES", 0.8), ("NO", 0.2)]),
        NoulQuestion("g", "Is it true?"),
        {"YES": ["YES", "Y"], "NO": ["NO", "N"]})
    assert a.type == "noul" and a.value == pytest.approx(0.8)

    # choice: an option outside top_logprobs gets the residual share
    a = backend._answer_from_logprobs(
        _lp([("A", 0.5), ("B", 0.3)]),
        ChoiceQuestion("c", "Pick", {"x": "X", "y": "Y", "z": "Z"}),
        {"A": ["A", "A)", "A."], "B": ["B", "B)", "B."], "C": ["C", "C)", "C."]})
    assert a.value == "x"
    assert a.probabilities == {"x": 0.5, "y": 0.3, "z": pytest.approx(0.2)}

    # score: value is the expected level index (.1*0 + .8*1 + .1*2)
    a = backend._answer_from_logprobs(
        _lp([("A", 0.1), ("B", 0.8), ("C", 0.1)]),
        ScoreQuestion("s", "Rate", ["none", "minor", "major"]),
        {"A": ["A"], "B": ["B"], "C": ["C"]})
    assert a.type == "score" and a.value == pytest.approx(1.0)


def test_llm_readout_greedy_fallback():
    from deciform.backends.llm import LLMAsJevBackend

    backend = LLMAsJevBackend(base_url="http://x", api_key=None, model="m")
    a = backend._answer_from_logprobs(
        {"message": {"content": "B"}, "logprobs": None},
        ChoiceQuestion("c", "Pick", {"x": "X", "y": "Y"}),
        ["A", "B"])
    assert a.value == "y"  # code B -> second option
    assert a.raw["mode"] == "greedy"


def test_debias_averages_option_probabilities():
    # 3 options, k=3: rot1 favors b, the rest favor a -> a wins the average
    class Varying(FakeBackend):
        def evaluate(self, state, questions):
            out = {}
            self.calls.append([q.name for q in questions])
            for q in questions:
                if q.name == "team#rot1":
                    out[q.name] = Answer(
                        type="choice", value="b",
                        probabilities={"a": 0.2, "b": 0.6, "c": 0.2})
                else:
                    out[q.name] = Answer(
                        type="choice", value="a",
                        probabilities={"a": 0.6, "b": 0.2, "c": 0.2})
            return out

    backend = Varying({})

    class M(BaseModel):
        team: Literal["a", "b", "c"]

    sf = Deciform(backend=backend, escalation=EscalationPolicy(debias_k=3))
    res = sf.extract(M, state="x")
    assert res.value.team == "a"
    # averaged over 3 rotations: a=(.6+.2+.6)/3, b=(.2+.6+.2)/3, c=.2
    assert res.fields[0].answer.probabilities == pytest.approx(
        {"a": 0.4667, "b": 0.3333, "c": 0.2}, abs=1e-3)
    assert len(backend.calls[0]) == 3  # 3 rotations in one batch

    # cap: k=3 on a 2-option field sends only 2 distinct rotations
    class M2(BaseModel):
        pick: Literal["x", "y"]

    backend2 = FakeBackend({})
    Deciform(backend=backend2, escalation=EscalationPolicy(debias_k=3)).extract(M2, state="x")
    assert len(backend2.calls[0]) == 2  # x,y and y,x; a third would double-weight
