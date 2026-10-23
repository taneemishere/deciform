"""Client for any POST /v1/systemone-compatible endpoint.

One implementation covers every provider that speaks the System One wire
protocol: Baseten's elephant-1/mammoth-1, TypeSafe Jev, Ollama,
system-one.dev, or your own gateway. Point ``base_url`` at the service and
pass the model slug — the class is deliberately provider-neutral.

    SystemOneBackend()                                   # defaults below
    SystemOneBackend(model=SystemOneBackend.MAMMOTH)
    SystemOneBackend(base_url="https://api.typesafe.ai", model="jev-latest",
                     api_key=os.environ["TYPESAFE_API_KEY"])

Key is read from ``SYSTEMONE_API_KEY``, falling back to ``BASETEN_API_KEY``
when the default endpoint is used.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from deciform.questions import Question, to_wire
from deciform.results import Answer

DEFAULT_BASE_URL = "https://inference.baseten.co"


def _parse_answer(body: dict[str, Any]) -> Answer:
    qtype = body.get("type")
    if qtype == "choice":
        return Answer(
            type="choice",
            value=body["choice"],
            probabilities=body.get("probabilities"),
            confidence=body.get("confidence"),
            raw=body,
        )
    if qtype == "noul":
        return Answer(type="noul", value=float(body["noul"]), raw=body)
    if qtype == "score":
        return Answer(
            type="score",
            value=body["score"],
            probabilities=body.get("probabilities"),
            confidence=body.get("confidence"),
            raw=body,
        )
    raise ValueError(f"Unknown answer type in System One response: {body!r}")


class SystemOneBackend:
    """DecisionBackend over a /v1/systemone endpoint."""

    # well-known open decision models (MIT-licensed, hosted on Baseten today)
    ELEPHANT = "decision-models/baseten-elephant-9b-latest"
    MAMMOTH = "decision-models/baseten-mammoth-27b-latest"

    def __init__(
        self,
        model: str = ELEPHANT,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        timeout: float = 60.0,
        name: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        api_key = api_key or os.environ.get("SYSTEMONE_API_KEY")
        if api_key is None and self.base_url == DEFAULT_BASE_URL:
            api_key = os.environ.get("BASETEN_API_KEY")
        self.name = name or f"systemone:{model.split('/')[-1]}"
        self.last_usage: dict[str, Any] | None = None
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    def evaluate(self, state: Any, questions: list[Question]) -> dict[str, Answer]:
        body = {
            "model": self.model,
            "state": state,
            "questions": {q.name: to_wire(q) for q in questions},
        }
        resp = self._client.post("/v1/systemone", json=body)
        resp.raise_for_status()
        payload = resp.json()
        self.last_usage = payload.get("usage")
        return {name: _parse_answer(a) for name, a in payload["answers"].items()}
