import logging
from collections.abc import Sequence
from typing import Any

import litellm
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from analytics_platform.config import Settings
from analytics_platform.llm.base import BaseLLM
from analytics_platform.llm.cache import LLMCache, make_cache_key
from analytics_platform.llm.types import LLMError, LLMResponse, Message, ModelTier

logger = logging.getLogger(__name__)

TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    RateLimitError,
    APIConnectionError,
    Timeout,
    ServiceUnavailableError,
    InternalServerError,
)


class LiteLLMClient(BaseLLM):
    """Calls the tier's model, retrying transient errors, then falls back to other models."""

    def __init__(self, settings: Settings, cache: LLMCache | None = None) -> None:
        self._models = {
            ModelTier.FAST: settings.llm_fast_model,
            ModelTier.STRONG: settings.llm_strong_model,
            ModelTier.JUDGE: settings.llm_judge_model,
        }
        self._fallbacks = settings.llm_fallback_models
        self._timeout_s = settings.llm_timeout_s
        self._max_retries = settings.llm_max_retries
        self._initial_wait_s = settings.llm_retry_initial_wait_s
        self._cache = cache

    async def _complete(
        self,
        messages: Sequence[Message],
        *,
        tier: ModelTier,
        json_mode: bool,
        temperature: float,
    ) -> LLMResponse:
        primary = self._models[tier]
        # Sampled (temperature > 0) outputs are intentionally varied, so they are never cached.
        cache = self._cache if temperature == 0 else None
        cache_key = make_cache_key(primary, messages, json_mode=json_mode)
        if cache is not None:
            hit = await cache.get(cache_key)
            if hit is not None:
                return hit.model_copy(update={"cached": True, "cost_usd": 0.0})

        candidates = [primary, *(m for m in self._fallbacks if m != primary)]
        errors: list[str] = []
        for model in candidates:
            try:
                response = await self._call_with_retry(model, messages, json_mode, temperature)
            except Exception as exc:
                logger.warning("LLM model %s failed: %s", model, exc)
                errors.append(f"{model}: {type(exc).__name__}: {exc}")
                continue
            if cache is not None:
                await cache.set(cache_key, response)
            return response
        raise LLMError("All models failed. " + " | ".join(errors))

    async def _call_with_retry(
        self,
        model: str,
        messages: Sequence[Message],
        json_mode: bool,
        temperature: float,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "timeout": self._timeout_s,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential_jitter(
                multiplier=self._initial_wait_s, jitter=self._initial_wait_s, max=20
            ),
            retry=retry_if_exception_type(TRANSIENT_ERRORS),
            reraise=True,
        ):
            with attempt:
                raw = await litellm.acompletion(**kwargs)
        return _to_response(raw, model)


def _to_response(raw: Any, requested_model: str) -> LLMResponse:
    usage = getattr(raw, "usage", None)
    try:
        cost = float(litellm.completion_cost(completion_response=raw))
    except Exception:
        cost = 0.0
    return LLMResponse(
        content=raw.choices[0].message.content or "",
        model=getattr(raw, "model", None) or requested_model,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cost_usd=cost,
    )
