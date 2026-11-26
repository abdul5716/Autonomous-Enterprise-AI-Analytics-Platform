from enum import StrEnum
from typing import Literal, TypedDict

from pydantic import BaseModel


class ModelTier(StrEnum):
    FAST = "fast"
    STRONG = "strong"
    JUDGE = "judge"


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False


class LLMError(Exception):
    pass


class StructuredOutputError(LLMError):
    pass
