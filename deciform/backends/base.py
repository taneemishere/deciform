"""Backend protocols — one interface for any decision-capable system.

Two backend kinds exist:

- ``DecisionBackend``: answers typed questions with probabilities, never
  generates text. Implementations: any ``/v1/systemone`` endpoint
  (``SystemOneBackend``), local encoder models, and the logprob readout
  over any chat LLM (``LLMAsJevBackend``).
- ``GenerativeBackend``: produces free-form values for fields no decision
  primitive can express (e.g. a ``str`` summary) and acts as the escalation
  target when a decision model is not confident enough.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from deciform.questions import Question
from deciform.results import Answer


@runtime_checkable
class DecisionBackend(Protocol):
    """Answers typed questions against a shared state with probabilities."""

    name: str

    def evaluate(self, state: Any, questions: list[Question]) -> dict[str, Answer]:
        """Answer every question in one batch over the same state.

        Implementations should send independent questions in a single request
        so the backend's shared-prefix caching applies.
        """
        ...


@runtime_checkable
class GenerativeBackend(Protocol):
    """Generates values for fields outside the decision primitives' reach."""

    name: str

    def produce(
        self,
        state: Any,
        instructions: str,
        context: dict[str, Any],
        schema_hint: dict[str, Any] | None = None,
    ) -> Any:
        """Return a single field value. ``context`` carries already-decided
        fields so dependent questions stay consistent. ``schema_hint`` is the
        leaf's JSON-schema fragment (type/enum/bounds) so the backend can
        constrain and validate its output."""
        ...
