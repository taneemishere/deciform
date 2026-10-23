"""Compiler: Pydantic model -> program of typed System One questions.

The heart of deciform. A schema field maps to a primitive based on what
a decision model can actually express:

    Literal / Enum ........ choice   (options = the members)
    bool .................. noul     (P(true))
    bounded int (span<=10)  score    (ordinal levels, rounded back)
    list[Literal] ......... one noul per member (multi-select)
    Optional[T] ........... presence noul gating T's question (round 2)
    nested BaseModel ...... recurse; leaves share the state prefix
    str / float / other ... generative fallback (no decision primitive exists)

Dependent questions form rounds: a field gated by an Optional presence check
is only asked if the presence noul clears 0.5.
"""

from __future__ import annotations

import enum
import types
from dataclasses import dataclass, field
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from deciform.questions import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion

MAX_SCORE_LEVELS = 10


@dataclass
class FieldPlan:
    """How one schema leaf maps to questions and back to a value."""

    path: str
    kind: str  # choice | noul | score | multinoul | generative
    question_names: list[str]
    annotation: Any
    value_map: dict[str, Any] = field(default_factory=dict)  # option key -> typed value
    bounds: tuple[int, int] | None = None  # for score fields
    default: Any = None
    gated_by: str | None = None  # presence-question that must fire first


@dataclass
class QuestionProgram:
    """A compiled schema: leaf plans plus the question set and dependencies."""

    fields: list[FieldPlan]
    questions: dict[str, Question]
    depends_on: dict[str, str]  # question name -> gating question name
    gate_roots: dict[str, tuple[str, Any]] = field(default_factory=dict)
    # presence-question name -> (optional field path, its declared default)


def _is_literal(ann: Any) -> bool:
    return get_origin(ann) is Literal


def _is_enum(ann: Any) -> bool:
    return isinstance(ann, type) and issubclass(ann, enum.Enum)


def _is_model(ann: Any) -> bool:
    return isinstance(ann, type) and issubclass(ann, BaseModel)


def _unwrap_optional(ann: Any) -> tuple[Any, bool]:
    """Return (inner, is_optional) for Optional[X] = Union[X, None]."""
    origin = get_origin(ann)
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(ann) if a is not type(None)]
        if len(args) == 1 and len(get_args(ann)) == 2:
            return args[0], True
    return ann, False


def _literal_options(ann: Any) -> list[Any]:
    """Flatten Literal[...] / Union[Literal, Literal] / Enum into value list."""
    if _is_literal(ann):
        return list(get_args(ann))
    origin = get_origin(ann)
    if origin in (Union, types.UnionType):
        values: list[Any] = []
        for arg in get_args(ann):
            if _is_literal(arg):
                values.extend(get_args(arg))
            elif _is_enum(arg):
                values.extend(e.value for e in arg)
            else:
                return []
        return values
    if _is_enum(ann):
        return [e.value for e in ann]
    return []


def _bounds(field_info: Any) -> tuple[int, int] | None:
    lo, hi = None, None
    for meta in getattr(field_info, "metadata", []):
        if getattr(meta, "ge", None) is not None:
            lo = meta.ge
        if getattr(meta, "le", None) is not None:
            hi = meta.le
    if lo is not None and hi is not None:
        return int(lo), int(hi)
    return None


def _title(name: str, field_info: Any) -> str:
    return getattr(field_info, "title", None) or name.replace("_", " ")


def _desc(field_info: Any) -> str:
    return getattr(field_info, "description", None) or ""


def _option_descriptions(field_info: Any) -> dict[str, str]:
    extra = getattr(field_info, "json_schema_extra", None) or {}
    return extra.get("options", {}) if isinstance(extra, dict) else {}


def _leaf_instructions(kind: str, path: str, title: str, desc: str) -> str:
    if desc:
        return desc
    where = f" for '{path}'" if "." in path else f" for '{title}'"
    if kind == "choice":
        return f"Select the best value{where}."
    if kind == "noul":
        return f"Is the statement for{where} true?"
    if kind == "score":
        return f"Rate the value{where} on the given scale."
    return f"Provide the value{where}."


def _walk(
    model: type[BaseModel],
    prefix: str,
    fields: list[FieldPlan],
    questions: dict[str, Question],
    depends_on: dict[str, str],
    gated_by: str | None,
    gate_roots: dict[str, tuple[str, Any]],
) -> None:
    hints = get_type_hints(model)  # resolves forward refs like Optional["Address"]
    for name, info in model.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        title = _title(name, info)
        desc = _desc(info)
        ann, optional = _unwrap_optional(hints.get(name, info.annotation))
        gate = gated_by

        if optional:
            presence = f"{path}__present"
            extra = getattr(info, "json_schema_extra", None) or {}
            gate_q = extra.get("gate_question") if isinstance(extra, dict) else None
            # Ask applicability, not literal presence: a record that *needs*
            # a reviewer rarely contains one, so "is X specified?" under-fires.
            questions[presence] = NoulQuestion(
                name=presence,
                instructions=gate_q
                or f"Should '{title}' be assigned a value for this record?",
            )
            if gated_by:
                depends_on[presence] = gated_by
            default = getattr(info, "default", None)
            gate_roots[presence] = (path, None if default is PydanticUndefined else default)
            gate = presence

        if _is_model(ann):
            _walk(ann, path, fields, questions, depends_on, gate, gate_roots)
            continue

        leaf = _make_leaf(path, title, desc, ann, info, questions)
        leaf.gated_by = gate  # generative leaves included: dead gate, no produce()
        fields.append(leaf)
        for q in leaf.question_names:
            if gate:
                depends_on[q] = gate


def _make_leaf(
    path: str,
    title: str,
    desc: str,
    ann: Any,
    info: Any,
    questions: dict[str, Question],
) -> FieldPlan:
    options = _literal_options(ann)
    if options:
        option_descs = _option_descriptions(info)
        criteria = {str(o): option_descs.get(str(o), str(o)) for o in options}
        questions[path] = ChoiceQuestion(
            name=path,
            instructions=_leaf_instructions("choice", path, title, desc),
            criteria=criteria,
        )
        return FieldPlan(
            path=path,
            kind="choice",
            question_names=[path],
            annotation=ann,
            value_map={str(o): o for o in options},
            default=getattr(info, "default", None),
        )

    if ann is bool:
        questions[path] = NoulQuestion(
            name=path,
            instructions=_leaf_instructions("noul", path, title, desc),
        )
        return FieldPlan(
            path=path, kind="noul", question_names=[path], annotation=ann,
            default=getattr(info, "default", None),
        )

    if ann is int:
        bounds = _bounds(info)
        if bounds and (bounds[1] - bounds[0] + 1) <= MAX_SCORE_LEVELS:
            lo, hi = bounds
            # numeric-only level names carry no semantics; callers can provide
            # per-level descriptions via Field(json_schema_extra={"level_labels": [...]})
            extra = getattr(info, "json_schema_extra", None) or {}
            labels = extra.get("level_labels") if isinstance(extra, dict) else None
            if labels and len(labels) == hi - lo + 1:
                criteria = [f"{i}: {labels[i - lo]}" for i in range(lo, hi + 1)]
            else:
                criteria = [str(i) for i in range(lo, hi + 1)]
            questions[path] = ScoreQuestion(
                name=path,
                instructions=_leaf_instructions("score", path, title, desc)
                or f"Rate '{title}' from {lo} to {hi}.",
                criteria=criteria,
            )
            return FieldPlan(
                path=path, kind="score", question_names=[path], annotation=ann,
                bounds=bounds, default=getattr(info, "default", None),
            )
        return _generative(path, title, desc, ann, info)

    if get_origin(ann) is list:
        (item_ann,) = get_args(ann) or (Any,)
        member_options = _literal_options(item_ann)
        if member_options:
            names = []
            for member in member_options:
                qname = f"{path}::{member}"
                questions[qname] = NoulQuestion(
                    name=qname,
                    instructions=f"Does the state's '{title}' include '{member}'?",
                )
                names.append(qname)
            return FieldPlan(
                path=path, kind="multinoul", question_names=names, annotation=ann,
                value_map={str(m): m for m in member_options},
                default=getattr(info, "default", None),
            )
        return _generative(path, title, desc, ann, info)

    return _generative(path, title, desc, ann, info)


def _generative(path: str, title: str, desc: str, ann: Any, info: Any) -> FieldPlan:
    return FieldPlan(
        path=path, kind="generative", question_names=[], annotation=ann,
        default=getattr(info, "default", None),
    )


def compile(model: type[BaseModel]) -> QuestionProgram:
    """Compile a Pydantic model into a question program."""
    fields: list[FieldPlan] = []
    questions: dict[str, Question] = {}
    depends_on: dict[str, str] = {}
    gate_roots: dict[str, tuple[str, Any]] = {}
    _walk(model, "", fields, questions, depends_on, None, gate_roots)
    return QuestionProgram(
        fields=fields, questions=questions, depends_on=depends_on, gate_roots=gate_roots
    )
