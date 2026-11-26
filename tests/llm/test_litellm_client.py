from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import litellm
import pytest
from litellm.exceptions import AuthenticationError, RateLimitError

from analytics_platform.config import Settings
from analytics_platform.llm import LiteLLMClient, LLMError, Message, ModelTier, track_usage
from analytics_platform.llm.cache import InMemoryLLMCache

USER_MSG: list[Message] = [{"role": "user", "content": "total revenue by region"}]


def fake_completion(content: str = "answer", model: str = "strong-model") -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class ScriptedLiteLLM:
    """Stands in for `litellm.acompletion`; `script` maps model name to a list of outcomes."""

    def __init__(self, script: dict[str, list[Any]]) -> None:
        self.script = script
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.script[kwargs["model"]].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def models_called(self) -> list[str]:
        return [c["model"] for c in self.calls]


Install = Callable[[dict[str, list[Any]]], ScriptedLiteLLM]


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> Install:
    def install(script: dict[str, list[Any]]) -> ScriptedLiteLLM:
        fake = ScriptedLiteLLM(script)
        monkeypatch.setattr(litellm, "acompletion", fake)
        return fake

    return install


def rate_limited() -> Exception:
    return RateLimitError(message="429", llm_provider="openai", model="strong-model")


def auth_failed() -> Exception:
    return AuthenticationError(message="bad key", llm_provider="openai", model="x")


async def test_routes_tier_to_configured_model(settings: Settings, scripted: Install) -> None:
    fake = scripted({"fast-model": [fake_completion(model="fast-model")]})

    reply = await LiteLLMClient(settings).complete(USER_MSG, tier=ModelTier.FAST)

    assert fake.models_called() == ["fast-model"]
    assert (reply.input_tokens, reply.output_tokens) == (10, 5)


async def test_transient_errors_are_retried(settings: Settings, scripted: Install) -> None:
    fake = scripted({"strong-model": [rate_limited(), rate_limited(), fake_completion()]})

    reply = await LiteLLMClient(settings).complete(USER_MSG)

    assert reply.content == "answer"
    assert fake.models_called() == ["strong-model"] * 3


async def test_falls_back_after_retries_exhausted(settings: Settings, scripted: Install) -> None:
    fake = scripted(
        {
            "strong-model": [rate_limited()] * 3,
            "fallback-model": [fake_completion("from fallback", model="fallback-model")],
        }
    )

    reply = await LiteLLMClient(settings).complete(USER_MSG)

    assert reply.content == "from fallback"
    assert fake.models_called() == ["strong-model"] * 3 + ["fallback-model"]


async def test_non_transient_error_skips_retry_and_falls_back(
    settings: Settings, scripted: Install
) -> None:
    fake = scripted({"strong-model": [auth_failed()], "fallback-model": [fake_completion()]})

    await LiteLLMClient(settings).complete(USER_MSG)

    assert fake.models_called() == ["strong-model", "fallback-model"]


async def test_raises_when_every_model_fails(settings: Settings, scripted: Install) -> None:
    scripted({"strong-model": [auth_failed()], "fallback-model": [auth_failed()]})

    with pytest.raises(LLMError, match="All models failed"):
        await LiteLLMClient(settings).complete(USER_MSG)


async def test_cache_hit_skips_provider_and_costs_nothing(
    settings: Settings, scripted: Install
) -> None:
    fake = scripted({"strong-model": [fake_completion()]})
    client = LiteLLMClient(settings, cache=InMemoryLLMCache())

    with track_usage() as usage:
        first = await client.complete(USER_MSG)
        second = await client.complete(USER_MSG)

    assert len(fake.calls) == 1
    assert (first.cached, second.cached) == (False, True)
    assert (usage.calls, usage.cached_calls) == (2, 1)


async def test_sampled_calls_bypass_cache(settings: Settings, scripted: Install) -> None:
    fake = scripted({"strong-model": [fake_completion(), fake_completion()]})
    client = LiteLLMClient(settings, cache=InMemoryLLMCache())

    await client.complete(USER_MSG, temperature=0.7)
    await client.complete(USER_MSG, temperature=0.7)

    assert len(fake.calls) == 2


async def test_json_mode_sets_response_format(settings: Settings, scripted: Install) -> None:
    fake = scripted({"strong-model": [fake_completion("{}")]})

    await LiteLLMClient(settings).complete(USER_MSG, json_mode=True)

    assert fake.calls[0]["response_format"] == {"type": "json_object"}
