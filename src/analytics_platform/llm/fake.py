import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from analytics_platform.llm.base import BaseLLM
from analytics_platform.llm.types import LLMError, LLMResponse, Message, ModelTier

Responder = str | dict[str, Any] | BaseModel | Callable[[Sequence[Message]], str]


@dataclass
class FakeCall:
    messages: list[Message]
    tier: ModelTier
    json_mode: bool
    response: str


@dataclass
class _Rule:
    pattern: re.Pattern[str]
    responder: Responder
    tier: ModelTier | None


@dataclass
class FakeLLM(BaseLLM):
    """Deterministic LLM for tests: the first rule whose regex matches the conversation wins.

    Unmatched prompts raise, so tests fail loudly instead of passing on a silent default.
    """

    default: Responder | None = None
    calls: list[FakeCall] = field(default_factory=list)
    _rules: list[_Rule] = field(default_factory=list)

    def when(
        self, pattern: str, responder: Responder, *, tier: ModelTier | None = None
    ) -> "FakeLLM":
        self._rules.append(_Rule(re.compile(pattern, re.IGNORECASE | re.DOTALL), responder, tier))
        return self

    async def _complete(
        self,
        messages: Sequence[Message],
        *,
        tier: ModelTier,
        json_mode: bool,
        temperature: float,
    ) -> LLMResponse:
        transcript = "\n".join(m["content"] for m in messages)
        responder = next(
            (
                r.responder
                for r in self._rules
                if (r.tier is None or r.tier == tier) and r.pattern.search(transcript)
            ),
            self.default,
        )
        if responder is None:
            raise LLMError(f"FakeLLM has no rule matching prompt:\n{transcript[-500:]}")
        content = _render(responder, messages)
        self.calls.append(FakeCall(list(messages), tier, json_mode, content))
        return LLMResponse(
            content=content,
            model="fake",
            input_tokens=len(transcript.split()),
            output_tokens=len(content.split()),
        )


def _render(responder: Responder, messages: Sequence[Message]) -> str:
    if isinstance(responder, BaseModel):
        return responder.model_dump_json()
    if isinstance(responder, dict):
        return json.dumps(responder)
    if isinstance(responder, str):
        return responder
    return responder(messages)
