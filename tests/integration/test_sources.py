"""Needs `make up` and `make data`. Run with: uv run pytest -m integration"""

import pytest

from analytics_platform.config import get_settings
from analytics_platform.llm import FakeLLM
from analytics_platform.security import DEMO_USERS
from analytics_platform.tools.base import ErrorKind, ToolError
from analytics_platform.tools.sql.tool import SQLTool

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tool() -> SQLTool:
    return SQLTool(FakeLLM(), get_settings())


async def test_postgres_row_level_security(tool: SQLTool) -> None:
    sql = "SELECT DISTINCT region FROM customers"

    everyone = await tool.run_sql("crm", sql, DEMO_USERS["finance.admin"])
    emea = await tool.run_sql("crm", sql, DEMO_USERS["emea.sales"])

    assert len(everyone.rows) == 4
    assert emea.rows == [{"region": "EMEA"}]


async def test_rls_applies_through_related_tables(tool: SQLTool) -> None:
    sql = "SELECT DISTINCT c.region FROM support_tickets t JOIN customers c USING (customer_id)"

    result = await tool.run_sql("crm", sql, DEMO_USERS["emea.sales"])

    assert result.rows == [{"region": "EMEA"}]


async def test_oracle_read_only_user_can_query_finance(tool: SQLTool) -> None:
    result = await tool.run_sql(
        "finance", "SELECT COUNT(*) AS n FROM refunds", DEMO_USERS["finance.admin"]
    )

    assert result.rows[0]["n"] > 0


async def test_database_errors_are_classified_for_repair(tool: SQLTool) -> None:
    with pytest.raises(ToolError) as info:
        await tool.run_sql(
            "crm", "SELECT no_such_column FROM customers", DEMO_USERS["finance.admin"]
        )
    assert info.value.kind == ErrorKind.INVALID_INPUT
