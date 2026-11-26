import hashlib
import logging
from typing import Protocol

from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


def cache_key(kind: str, value: str) -> str:
    return f"web:v1:{kind}:" + hashlib.sha256(value.encode()).hexdigest()


class TextCache(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str) -> None: ...


class InMemoryTextCache:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str) -> None:
        self.store[key] = value


class RedisTextCache:
    """Degrades to a cache miss when Redis is unavailable."""

    def __init__(self, url: str, ttl_s: int) -> None:
        self._client = Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
        self._ttl_s = ttl_s

    async def get(self, key: str) -> str | None:
        try:
            raw = await self._client.get(key)
        except RedisError as exc:
            logger.warning("Web cache read failed, treating as miss: %s", exc)
            return None
        return raw.decode() if isinstance(raw, bytes) else raw

    async def set(self, key: str, value: str) -> None:
        try:
            await self._client.set(key, value, ex=self._ttl_s)
        except RedisError as exc:
            logger.warning("Web cache write failed: %s", exc)
