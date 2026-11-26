import json
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from analytics_platform.llm.types import (
    LLMResponse,
    Message,
    ModelTier,
    StructuredOutputError,
)
from analytics_platform.llm.usage import record_usage

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def strip_code_fences(text: str) -> str:
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text


class BaseLLM(ABC):
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tier: ModelTier = ModelTier.STRONG,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> LLMResponse:
        response = await self._complete(
            messages, tier=tier, json_mode=json_mode, temperature=temperature
        )
        record_usage(response)
        return response

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        tier: ModelTier = ModelTier.STRONG,
        max_repairs: int = 1,
    ) -> T:
        """Return a validated `schema` instance, feeding validation errors back for repair."""
        instruction: Message = {
            "role": "system",
            "content": (
                "Respond with only a JSON object that conforms to this JSON Schema:\n"
                + json.dumps(schema.model_json_schema())
            ),
        }
        conversation: list[Message] = [instruction, *messages]
        for attempt in range(max_repairs + 1):
            response = await self.complete(conversation, tier=tier, json_mode=True)
            try:
                return schema.model_validate_json(strip_code_fences(response.content))
            except ValidationError as exc:
                if attempt == max_repairs:
                    raise StructuredOutputError(
                        f"{schema.__name__} output still invalid after {max_repairs} repairs: {exc}"
                    ) from exc
                conversation = [
                    *conversation,
                    {"role": "assistant", "content": response.content},
                    {
                        "role": "user",
                        "content": f"That JSON was invalid:\n{exc}\nReturn corrected JSON only.",
                    },
                ]
        raise AssertionError("unreachable")

    @abstractmethod
    async def _complete(
        self,
        messages: Sequence[Message],
        *,
        tier: ModelTier,
        json_mode: bool,
        temperature: float,
    ) -> LLMResponse: ...
