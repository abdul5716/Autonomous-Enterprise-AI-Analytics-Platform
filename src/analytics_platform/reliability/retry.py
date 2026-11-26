"""Retries with exponential backoff and jitter, for classified tool errors only."""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable

from analytics_platform.llm import LLMError
from analytics_platform.tools.base import ErrorKind, ToolError

logger = logging.getLogger(__name__)


def as_tool_error(exc: Exception) -> ToolError:
    """Classify any failure so the agent can decide what to do with it."""
    if isinstance(exc, ToolError):
        return exc
    if isinstance(exc, LLMError):
        # The LLM client has already retried and tried its fallback models.
        return ToolError(
            f"The language model failed: {exc}", kind=ErrorKind.UNAVAILABLE, source="llm"
        )
    if isinstance(exc, TimeoutError):
        return ToolError("The step timed out.", kind=ErrorKind.TIMEOUT)
    logger.exception("Unclassified tool failure", exc_info=exc)
    return ToolError(f"Internal error: {type(exc).__name__}: {exc}", kind=ErrorKind.UNAVAILABLE)


async def retry_async[T](
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base_delay_s: float,
    on_retry: Callable[[int, ToolError], None] | None = None,
) -> T:
    """Call `fn`, retrying transient errors and timeouts; anything else is raised at once."""
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except Exception as raw:
            exc = as_tool_error(raw)
            if not exc.retryable or attempt == attempts:
                raise exc from raw
            if on_retry:
                on_retry(attempt, exc)
            delay = base_delay_s * 2 ** (attempt - 1)
            await asyncio.sleep(delay + random.uniform(0, delay / 2))
    raise AssertionError("unreachable")
