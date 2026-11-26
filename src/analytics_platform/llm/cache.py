import hashlib
import json
import logging
from collections.abc import Sequence
from typing import Protocol

from redis.asyncio import Redis
from redis.exceptions import RedisError

from analytics_platform.llm.types import LLMResponse, Message

logger = logging.getLogger(__name__)


def make_cache_key(model: str, messages: Sequence[Message], *, json_mode: bool) -> str:
    payload = json.dumps(
        {"model": model, "messages": list(messages), "json_mode": json_mode}, sort_keys=True
    )
    return "llm:v1:" + hashlib.sha256(payload.encode()).hexdigest()


class LLMCache(Protocol):
    async def get(self, key: str) -> LLMResponse | None: ...

    async def set(self, key: str, response: LLMResponse) -> None: ...


class InMemoryLLMCache:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, key: str) -> LLMResponse | None:
        raw = self._store.get(key)
        return LLMResponse.model_validate_json(raw) if raw else None

    async def set(self, key: str, response: LLMResponse) -> None:
        self._store[key] = response.model_dump_json()


class RedisLLMCache:
    """Redis-backed cache that degrades to a cache miss when Redis is unavailable."""

    def __init__(self, client: Redis, ttl_s: int) -> None:
        self._client = client
        self._ttl_s = ttl_s

    @classmethod
    def from_url(cls, url: str, ttl_s: int) -> "RedisLLMCache":
        return cls(Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2), ttl_s)

    async def get(self, key: str) -> LLMResponse | None:
        try:
            raw = await self._client.get(key)
        except RedisError as exc:
            logger.warning("LLM cache read failed, treating as miss: %s", exc)
            return None
        return LLMResponse.model_validate_json(raw) if raw else None

    async def set(self, key: str, response: LLMResponse) -> None:
        try:
            await self._client.set(key, response.model_dump_json(), ex=self._ttl_s)
        except RedisError as exc:
            logger.warning("LLM cache write failed: %s", exc)
