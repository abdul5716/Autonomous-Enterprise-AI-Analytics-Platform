import pytest
from pydantic import BaseModel

from analytics_platform.llm import FakeLLM, StructuredOutputError
from analytics_platform.llm.base import strip_code_fences


class Intent(BaseModel):
    goal: str
    needs_sql: bool


async def test_invalid_json_is_repaired_with_error_feedback() -> None:
    llm = FakeLLM().when("That JSON was invalid", '{"goal": "revenue", "needs_sql": true}')
    llm.when("analyze", '{"goal": "revenue"}')

    result = await llm.structured([{"role": "user", "content": "analyze"}], Intent)

    assert result == Intent(goal="revenue", needs_sql=True)
    assert len(llm.calls) == 2
    assert "needs_sql" in llm.calls[1].messages[-1]["content"]


async def test_gives_up_after_max_repairs() -> None:
    llm = FakeLLM(default="not json at all")

    with pytest.raises(StructuredOutputError, match="Intent"):
        await llm.structured([{"role": "user", "content": "x"}], Intent, max_repairs=2)
    assert len(llm.calls) == 3


async def test_schema_is_sent_to_the_model() -> None:
    llm = FakeLLM(default='{"goal": "g", "needs_sql": false}')

    await llm.structured([{"role": "user", "content": "x"}], Intent)

    assert '"needs_sql"' in llm.calls[0].messages[0]["content"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        ('{"a": 1}', '{"a": 1}'),
    ],
)
def test_strip_code_fences(raw: str, expected: str) -> None:
    assert strip_code_fences(raw) == expected
