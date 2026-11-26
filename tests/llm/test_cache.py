from fakeredis import FakeAsyncRedis
from redis.exceptions import ConnectionError as RedisConnectionError

from analytics_platform.llm import LLMResponse, Message
from analytics_platform.llm.cache import InMemoryLLMCache, RedisLLMCache, make_cache_key

RESPONSE = LLMResponse(content="hi", model="m", input_tokens=3, output_tokens=1, cost_usd=0.01)


def test_cache_key_depends_on_model_messages_and_mode() -> None:
    msgs: list[Message] = [{"role": "user", "content": "q"}]
    base = make_cache_key("a", msgs, json_mode=False)

    assert base == make_cache_key("a", msgs, json_mode=False)
    assert base != make_cache_key("b", msgs, json_mode=False)
    assert base != make_cache_key("a", msgs, json_mode=True)


async def test_redis_cache_roundtrip_with_ttl() -> None:
    client = FakeAsyncRedis()
    cache = RedisLLMCache(client, ttl_s=60)

    await cache.set("k", RESPONSE)

    assert await cache.get("k") == RESPONSE
    assert 0 < await client.ttl("k") <= 60


async def test_redis_outage_degrades_to_miss() -> None:
    class BrokenRedis:
        async def get(self, key: str) -> bytes:
            raise RedisConnectionError("down")

        async def set(self, *args: object, **kwargs: object) -> None:
            raise RedisConnectionError("down")

    cache = RedisLLMCache(BrokenRedis(), ttl_s=60)  # type: ignore[arg-type]

    await cache.set("k", RESPONSE)
    assert await cache.get("k") is None


async def test_in_memory_cache_roundtrip() -> None:
    cache = InMemoryLLMCache()

    assert await cache.get("k") is None
    await cache.set("k", RESPONSE)
    assert await cache.get("k") == RESPONSE
