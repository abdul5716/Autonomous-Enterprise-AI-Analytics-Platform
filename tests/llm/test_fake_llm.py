import pytest
from pydantic import BaseModel

from analytics_platform.llm import FakeLLM, LLMError, ModelTier, track_usage


class Route(BaseModel):
    route: str
    confidence: float


async def test_first_matching_rule_wins() -> None:
    llm = FakeLLM().when("refund policy", "rag").when("revenue", "sql")

    reply = await llm.complete([{"role": "user", "content": "Show total revenue by region"}])

    assert reply.content == "sql"
    assert llm.calls[0].tier == ModelTier.STRONG


async def test_rules_can_be_scoped_to_a_tier() -> None:
    llm = (
        FakeLLM()
        .when("question", "fast answer", tier=ModelTier.FAST)
        .when("question", "strong answer")
    )

    fast = await llm.complete([{"role": "user", "content": "question"}], tier=ModelTier.FAST)
    strong = await llm.complete([{"role": "user", "content": "question"}])

    assert (fast.content, strong.content) == ("fast answer", "strong answer")


async def test_unmatched_prompt_raises() -> None:
    with pytest.raises(LLMError, match="no rule matching"):
        await FakeLLM().complete([{"role": "user", "content": "anything"}])


async def test_structured_output_from_model_responder() -> None:
    llm = FakeLLM().when("classify", Route(route="sql", confidence=0.9))

    result = await llm.structured([{"role": "user", "content": "classify this"}], Route)

    assert result == Route(route="sql", confidence=0.9)
    assert llm.calls[0].json_mode is True


async def test_usage_is_tracked_per_scope() -> None:
    llm = FakeLLM(default="ok")

    with track_usage() as usage:
        await llm.complete([{"role": "user", "content": "one"}])
        await llm.complete([{"role": "user", "content": "two"}])
    await llm.complete([{"role": "user", "content": "outside"}])

    assert usage.calls == 2
