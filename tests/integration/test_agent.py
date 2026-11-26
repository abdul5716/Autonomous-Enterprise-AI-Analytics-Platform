"""The agent against the real warehouse, with checkpoints in Postgres (scripted LLM)."""

import json
from typing import Any

import pytest

from analytics_platform.agent.checkpoint import postgres_checkpointer
from analytics_platform.agent.graph import build_graph
from analytics_platform.agent.service import AgentService
from analytics_platform.agent.toolbox import Toolbox
from analytics_platform.config import Settings
from analytics_platform.llm import FakeLLM
from analytics_platform.security import DEMO_USERS
from analytics_platform.tools.sql.tool import SQLTool
from tests.agent.conftest import INTENT, SYNTHESIS, draft, intent

pytestmark = pytest.mark.integration

REVENUE_BY_REGION = {
    "answerable": True,
    "reason": "governed metric",
    "query": {
        "metrics": ["net_revenue"],
        "dimensions": ["region"],
        "time_range": {"period": "last_quarter"},
        "compare_to": "previous_period",
    },
}


def scripted_llm() -> FakeLLM:
    def classify(messages: Any) -> str:
        if "Clarification - Q:" in str(messages):
            return json.dumps(intent("data_query", "table", goal="Net revenue by region"))
        return json.dumps(
            intent(
                "data_query",
                ambiguity="'performance' is not a defined metric",
                question="Which metric: revenue, refunds or customers?",
            )
        )

    return (
        FakeLLM()
        .when(INTENT, classify)
        .when("governed metric query", REVENUE_BY_REGION)
        .when(SYNTHESIS, draft("Revenue by region [S1].", {"C1": ["S1"]}))
    )


def toolbox(settings: Settings) -> Toolbox:
    llm = scripted_llm()
    return Toolbox(llm=llm, settings=settings, sql=SQLTool(llm, settings))


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, llm_provider="fake", agent_retry_base_delay_s=0)


async def test_clarification_resumes_from_postgres_after_a_restart(settings: Settings) -> None:
    async with postgres_checkpointer(settings) as saver:
        first = await AgentService(build_graph(toolbox(settings), saver)).ask(
            "How is performance by region?", DEMO_USERS["finance.admin"]
        )
    assert first.status == "needs_clarification"

    async with postgres_checkpointer(settings) as saver:  # a new process, same thread id
        second = await AgentService(build_graph(toolbox(settings), saver)).clarify(
            first.thread_id, "revenue"
        )

    answer = second.answer
    assert answer is not None and answer.status == "complete"
    table = answer.tables[0]
    change = dict(
        zip(
            [r[0] for r in table.rows],
            [r[table.columns.index("net_revenue_change_pct")] for r in table.rows],
            strict=True,
        )
    )
    assert change["EMEA"] == pytest.approx(-28.71) and change["APAC"] == pytest.approx(-33.99)


async def test_regional_user_only_gets_their_region(settings: Settings) -> None:
    async with postgres_checkpointer(settings) as saver:
        service = AgentService(build_graph(toolbox(settings), saver))
        first = await service.ask("How is performance by region?", DEMO_USERS["emea.sales"])
        result = await service.clarify(first.thread_id, "revenue")

    answer = result.answer
    assert answer is not None
    assert [row[0] for row in answer.tables[0].rows] == ["EMEA"]
    assert "Results are limited to your regions: EMEA." in answer.assumptions
