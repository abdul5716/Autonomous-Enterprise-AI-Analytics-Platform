import pytest

from analytics_platform.config import Settings
from analytics_platform.llm import FakeLLM
from analytics_platform.security import DEMO_USERS
from analytics_platform.tools.base import ErrorKind, ToolError
from analytics_platform.tools.sql.semantic import MetricQuery
from analytics_platform.tools.sql.sources import DuckDBWarehouse
from analytics_platform.tools.sql.tool import SQLTool

ADMIN = DEMO_USERS["finance.admin"]
EMEA = DEMO_USERS["emea.sales"]


def make_tool(settings: Settings, llm: FakeLLM | None = None) -> SQLTool:
    return SQLTool(llm or FakeLLM(), settings, sources={"warehouse": DuckDBWarehouse(settings)})


def plan(query: dict[str, object] | None, answerable: bool = True) -> dict[str, object]:
    return {"answerable": answerable, "reason": "test", "query": query}


async def test_metric_query_with_comparison(warehouse_settings: Settings) -> None:
    tool = make_tool(warehouse_settings)
    q = MetricQuery.model_validate(
        {
            "metrics": ["net_revenue"],
            "dimensions": ["region"],
            "time_range": {"period": "last_quarter"},
            "compare_to": "previous_period",
        }
    )

    result = await tool.run_metric_query(q, ADMIN)

    rows = {r["region"]: r for r in result.rows}
    assert rows["NA"]["net_revenue"] == 900.0
    assert rows["NA"]["net_revenue_change_pct"] == -10.0
    assert rows["EMEA"]["net_revenue"] == 250.0  # 300 invoiced minus a 50 SLA credit
    assert rows["EMEA"]["net_revenue_change_pct"] == -50.0
    assert result.comparison_period is not None
    assert result.comparison_period.label == "2026-Q2"
    assert result.citation.source_type == "sql"


async def test_region_scope_limits_rows_and_is_disclosed(warehouse_settings: Settings) -> None:
    q = MetricQuery.model_validate({"metrics": ["net_revenue"], "dimensions": ["region"]})

    result = await make_tool(warehouse_settings).run_metric_query(q, EMEA)

    assert [r["region"] for r in result.rows] == ["EMEA"]
    assert any("limited to your regions: EMEA" in a for a in result.assumptions)


async def test_compare_with_time_dimension_is_rejected(warehouse_settings: Settings) -> None:
    q = MetricQuery.model_validate(
        {
            "metrics": ["net_revenue"],
            "dimensions": ["month"],
            "time_range": {"period": "last_quarter"},
            "compare_to": "previous_period",
        }
    )

    with pytest.raises(ToolError, match="compare_to cannot be combined") as info:
        await make_tool(warehouse_settings).run_metric_query(q, ADMIN)
    assert info.value.kind == ErrorKind.INVALID_INPUT


async def test_ask_uses_semantic_layer(warehouse_settings: Settings) -> None:
    llm = FakeLLM().when(
        "governed metric query",
        plan(
            {
                "metrics": ["net_revenue", "budget"],
                "dimensions": ["region"],
                "time_range": {"period": "last_quarter"},
            }
        ),
    )

    result = await make_tool(warehouse_settings, llm).ask("Revenue vs budget last quarter", ADMIN)

    assert result.method == "semantic"
    assert {r["region"]: r["budget"] for r in result.rows} == {"NA": 1100.0, "EMEA": 600.0}


async def test_ask_repairs_invalid_semantic_query(warehouse_settings: Settings) -> None:
    llm = (
        FakeLLM()
        .when("Fix these problems", plan({"metrics": ["net_revenue"], "dimensions": ["region"]}))
        .when("governed metric query", plan({"metrics": ["revenue"], "dimensions": ["region"]}))
    )

    result = await make_tool(warehouse_settings, llm).ask("Revenue by region", ADMIN)

    assert result.method == "semantic"
    assert "Unknown metric 'revenue'" in llm.calls[1].messages[-1]["content"]


async def test_ask_falls_back_to_sql_and_repairs_it(warehouse_settings: Settings) -> None:
    llm = (
        FakeLLM()
        .when("governed metric query", plan(None, answerable=False))
        .when(
            "The query failed",
            {
                "source": "warehouse",
                "explanation": "fixed",
                "sql": "SELECT refund_reason, SUM(-amount_usd) AS total FROM fact_revenue "
                "WHERE txn_type = 'refund' GROUP BY 1",
            },
        )
        .when(
            "read-only SQL query",
            {"source": "warehouse", "explanation": "first try", "sql": "SELECT * FROM refunds"},
        )
    )

    result = await make_tool(warehouse_settings, llm).ask("Refunds by reason", ADMIN)

    assert result.method == "sql"
    assert result.rows == [{"refund_reason": "sla_credit", "total": 50.0}]
    assert "Table 'refunds' is not accessible" in llm.calls[-1].messages[-1]["content"]


async def test_custom_sql_on_warehouse_is_denied_for_region_restricted_users(
    warehouse_settings: Settings,
) -> None:
    tool = make_tool(warehouse_settings)

    with pytest.raises(ToolError) as info:
        await tool.run_sql("warehouse", "SELECT * FROM fact_revenue", EMEA)
    assert info.value.kind == ErrorKind.PERMISSION
