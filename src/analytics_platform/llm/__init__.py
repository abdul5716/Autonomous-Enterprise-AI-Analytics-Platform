from analytics_platform.config import Settings, get_settings
from analytics_platform.llm.base import BaseLLM
from analytics_platform.llm.cache import RedisLLMCache
from analytics_platform.llm.fake import FakeLLM
from analytics_platform.llm.litellm_client import LiteLLMClient
from analytics_platform.llm.types import (
    LLMError,
    LLMResponse,
    Message,
    ModelTier,
    StructuredOutputError,
)
from analytics_platform.llm.usage import Usage, track_usage

__all__ = [
    "BaseLLM",
    "FakeLLM",
    "LLMError",
    "LLMResponse",
    "LiteLLMClient",
    "Message",
    "ModelTier",
    "StructuredOutputError",
    "Usage",
    "build_llm",
    "track_usage",
]


def build_llm(settings: Settings | None = None) -> BaseLLM:
    settings = settings or get_settings()
    if settings.llm_provider == "fake":
        return FakeLLM()
    cache = (
        RedisLLMCache.from_url(settings.redis_url, settings.llm_cache_ttl_s)
        if settings.llm_cache_enabled
        else None
    )
    return LiteLLMClient(settings, cache)
