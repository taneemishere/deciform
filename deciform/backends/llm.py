"""LLM-side backends for any OpenAI-compatible chat-completions API.

- ``OpenAIGenerativeBackend``: GenerativeBackend for free-text fields and
  escalations — one JSON call per field, schema-hinted and validated.
- ``LLMAsJevBackend``: DecisionBackend that reads first-token top_logprobs
  over option letters — the training-free "LLM-as-Jev" readout. Turns any
  chat-completions endpoint with logprobs into a decision model; falls back
  to a greedy one-hot answer when logprobs are unavailable.

Works with any OpenAI-compatible endpoint; reasoning models get
``chat_template_kwargs={"enable_thinking": false}`` automatically (the
flag is dropped and retried on endpoints that reject it).
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from typing import Any

import httpx

from deciform.questions import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion
from deciform.results import Answer

def _option_codes(n: int) -> list[str]:
    """A, B, ..., Z, AA, AB, ... — supports any option count."""
    codes = []
    for i in range(n):
        if i < 26:
            codes.append(_LETTERS[i])
        else:
            hi, lo = divmod(i - 26, 26)
            codes.append(_LETTERS[hi] + _LETTERS[lo])
    return codes


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class _ChatClient:
    """Minimal OpenAI-compatible chat client with optional RPM spacing."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: float = 120.0,
        requests_per_minute: float | None = None,
        disable_thinking: bool = True,
    ):
        self.model = model
        self.disable_thinking = disable_thinking
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._last_call = 0.0
        self._lock = threading.Lock()
        self.last_usage: dict[str, Any] | None = None
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    def _throttle(self) -> None:
        if not self._min_interval:
            return
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last_call)
            self._last_call = time.monotonic() + wait
        if wait > 0:
            time.sleep(wait)

    def chat(self, messages: list[dict], **kwargs: Any) -> dict:
        body: dict[str, Any] = {"model": self.model, "messages": messages, **kwargs}
        if self.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        for attempt in range(5):
            self._throttle()
            resp = self._client.post("/chat/completions", json=body)
            if resp.status_code == 400 and "chat_template_kwargs" in body:
                # Endpoint doesn't know chat_template_kwargs -> drop it for good.
                self.disable_thinking = False
                body.pop("chat_template_kwargs", None)
                continue
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                ra = resp.headers.get("retry-after")
                try:
                    wait = float(ra) if ra else 2.0 * (2 ** attempt)
                except ValueError:
                    wait = 2.0 * (2 ** attempt)
                time.sleep(min(wait, 60.0))
                continue
            resp.raise_for_status()
            data = resp.json()
            self.last_usage = data.get("usage")
            return data["choices"][0]


def _state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    try:
        return json.dumps(state, indent=2, default=str)
    except TypeError:
        return str(state)


class OpenAIGenerativeBackend:
    """GenerativeBackend over any OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        name: str | None = None,
        **client_kwargs: Any,
    ):
        self._client = _ChatClient(base_url, api_key, model, **client_kwargs)
        self.name = name or model

    @property
    def last_usage(self) -> dict[str, Any] | None:
        return self._client.last_usage

    def produce(
        self,
        state: Any,
        instructions: str,
        context: dict[str, Any],
        schema_hint: dict[str, Any] | None = None,
    ) -> Any:
        prompt = "You fill in one field of a structured decision object.\n\n"
        prompt += f"State:\n{_state_text(state)}\n\n"
        if context:
            prompt += f"Fields already decided:\n{json.dumps(context, default=str)}\n\n"
        prompt += f"Task: {instructions}\n"
        if schema_hint:
            prompt += f"The value must match this JSON schema: {json.dumps(schema_hint)}\n"
        prompt += 'Respond with a JSON object exactly {"value": <the value>} and nothing else.'

        messages = [{"role": "user", "content": prompt}]
        choice = self._client.chat(messages, max_tokens=128, temperature=0.0)
        value = _parse_value(choice["message"].get("content") or "")

        allowed = (schema_hint or {}).get("enum")
        if allowed and value not in allowed:
            # one cheap repair pass before giving up
            messages.append({"role": "assistant", "content": choice["message"]["content"]})
            messages.append({
                "role": "user",
                "content": f"Invalid. Choose exactly one of {allowed}. "
                'Respond only with {"value": ...}.',
            })
            retry = self._client.chat(messages, max_tokens=64, temperature=0.0)
            value = _parse_value(retry["message"].get("content") or "")
        return value


def _parse_value(text: str) -> Any:
    """Pull {"value": X} out of a model response; bare text falls through."""
    text = text.strip()
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))["value"]
        except (json.JSONDecodeError, KeyError):
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text.strip('"')


class LLMAsJevBackend:
    """DecisionBackend via first-token logprobs over option letters.

    Each question is one chat call with max_tokens small; the answer is read
    from the token distribution instead of parsed text, so a hosted LLM can
    stand in for a purpose-built decision model (arXiv 2610.02076).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        name: str | None = None,
        **client_kwargs: Any,
    ):
        self._client = _ChatClient(base_url, api_key, model, **client_kwargs)
        self.name = name or f"llm-as-jev:{model}"

    @property
    def last_usage(self) -> dict[str, Any] | None:
        return self._client.last_usage

    def evaluate(self, state: Any, questions: list[Question]) -> dict[str, Answer]:
        return {q.name: self._ask(state, q) for q in questions}

    def _ask(self, state: Any, q: Question) -> Answer:
        if isinstance(q, NoulQuestion):
            prompt = (
                f"State:\n{_state_text(state)}\n\n"
                f"Statement: {q.instructions}\n\n"
                "Answer with only YES or NO."
            )
            targets: dict[str, list[str]] = {"YES": ["YES", "Y"], "NO": ["NO", "N"]}
        else:
            criteria = q.criteria if isinstance(q, (ChoiceQuestion, ScoreQuestion)) else {}
            descs = list(criteria.values()) if isinstance(criteria, dict) else list(criteria)
            codes = _option_codes(len(descs))
            lines = "\n".join(f"{c}) {d}" for c, d in zip(codes, descs))
            prompt = (
                f"State:\n{_state_text(state)}\n\n"
                f"Question: {q.instructions}\n\nOptions:\n{lines}\n\n"
                "Answer with only the code of the best option."
            )
            targets = {c: [c, f"{c})", f"{c}."] for c in codes}

        # A few tokens: models don't always put the answer at position 0, so we
        # scan positions for the one where option-letter mass concentrates.
        choice = self._client.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=6, temperature=0.0, logprobs=True,
            top_logprobs=min(20, max(len(targets) + 2, 5)),
        )
        return self._answer_from_logprobs(choice, q, targets)

    def _answer_from_logprobs(self, choice: dict, q: Question,
                              targets: dict[str, list[str]]) -> Answer:
        content = (choice.get("logprobs") or {}).get("content") or []
        if not content:
            return self._greedy_answer(choice, q, list(targets))

        all_aliases = {a for aliases in targets.values() for a in aliases}
        best_entries, best_mass = None, 0.0
        for pos in content[:6]:
            # (normalized token, probability) — variants like "YES"/"Yes"/"yes"
            # all count toward the same target, so keep them as a list to SUM.
            entries = [
                (e["token"].strip().upper(), math.exp(e["logprob"]))
                for e in pos.get("top_logprobs", [])
            ]
            mass = sum(p for tok, p in entries if tok in all_aliases)
            if mass > best_mass:
                best_mass, best_entries = mass, entries
        if not best_entries:
            return self._greedy_answer(choice, q, list(targets))

        raw: dict[str, float] = {}
        for target, aliases in targets.items():
            total = sum(p for tok, p in best_entries if tok in aliases)
            if total > 0:
                raw[target] = total
        if not raw:
            return self._greedy_answer(choice, q, list(targets))

        total_seen = sum(raw.values())
        residual = max(0.0, 1.0 - total_seen)
        missing = [t for t in targets if t not in raw]
        probs = {t: raw.get(t, residual / max(1, len(missing))) for t in targets}

        if isinstance(q, NoulQuestion):
            p_yes = probs.get("YES", 0.0)
            denom = p_yes + probs.get("NO", 0.0)
            value = p_yes / denom if denom else 0.5
            return Answer(type="noul", value=value,
                          raw={"letter_probs": probs, "mode": "logprobs"})

        options = list(q.criteria) if isinstance(q, ChoiceQuestion) else [str(i) for i in range(len(q.criteria))]
        codes = _option_codes(len(options))
        option_probs = {opt: probs.get(codes[i], 0.0) for i, opt in enumerate(options)}
        if isinstance(q, ChoiceQuestion):
            best = max(option_probs, key=option_probs.get)
            return Answer(type="choice", value=best, probabilities=option_probs,
                          confidence=option_probs[best], raw={"mode": "logprobs"})
        expected = sum(i * p for i, p in enumerate(option_probs.values()))
        return Answer(type="score", value=expected, probabilities=option_probs,
                      confidence=max(option_probs.values()), raw={"mode": "logprobs"})

    def _greedy_answer(self, choice: dict, q: Question, targets: list[str]) -> Answer:
        """Parse the generated text when logprobs are unavailable."""
        text = ((choice.get("message") or {}).get("content") or "").strip().upper()
        if isinstance(q, NoulQuestion):
            return Answer(type="noul", value=0.9 if text.startswith("YES") else 0.1,
                          raw={"mode": "greedy", "text": text})
        code = next(
            (t for t in sorted(targets, key=len, reverse=True)
             if re.search(rf"\b{re.escape(t)}\b", text)),
            targets[0],
        )
        idx = targets.index(code)
        if isinstance(q, ChoiceQuestion):
            options = list(q.criteria)
            return Answer(type="choice", value=options[idx],
                          probabilities={o: (1.0 if o == options[idx] else 0.0) for o in options},
                          confidence=1.0, raw={"mode": "greedy", "text": text})
        return Answer(type="score", value=float(idx),
                      probabilities={str(i): (1.0 if i == idx else 0.0) for i in range(len(targets))},
                      confidence=1.0, raw={"mode": "greedy", "text": text})
