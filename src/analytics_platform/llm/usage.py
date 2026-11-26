from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from analytics_platform.llm.types import LLMResponse


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, response: LLMResponse) -> None:
        self.calls += 1
        self.cached_calls += int(response.cached)
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.cost_usd += response.cost_usd


_current_usage: ContextVar[Usage | None] = ContextVar("llm_usage", default=None)


@contextmanager
def track_usage() -> Iterator[Usage]:
    """Collect LLM usage for everything awaited inside the block, isolated per async task."""
    usage = Usage()
    token = _current_usage.set(usage)
    try:
        yield usage
    finally:
        _current_usage.reset(token)


def record_usage(response: LLMResponse) -> None:
    usage = _current_usage.get()
    if usage is not None:
        usage.add(response)
