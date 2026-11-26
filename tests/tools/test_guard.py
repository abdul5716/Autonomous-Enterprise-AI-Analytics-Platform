import pytest

from analytics_platform.tools.base import ErrorKind, ToolError
from analytics_platform.tools.sql.guard import guard_sql

CRM = {"allowed_tables": {"customers", "subscriptions"}, "allowed_schemas": {"crm"}}


def guard(sql: str, dialect: str = "postgres", max_rows: int = 100) -> str:
    return guard_sql(sql, dialect=dialect, max_rows=max_rows, **CRM).sql


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT * FROM customers", "SELECT * FROM customers LIMIT 101"),
        ("SELECT * FROM crm.customers LIMIT 5", "SELECT * FROM crm.customers LIMIT 5"),
        ("SELECT * FROM customers LIMIT 5000", "SELECT * FROM customers LIMIT 101"),
        (
            "WITH c AS (SELECT region FROM customers) SELECT * FROM c",
            "WITH c AS (SELECT region FROM customers) SELECT * FROM c LIMIT 101",
        ),
    ],
)
def test_allows_reads_and_caps_rows(sql: str, expected: str) -> None:
    assert guard(sql) == expected


def test_unions_are_wrapped_with_a_limit() -> None:
    sql = guard("SELECT region FROM customers UNION SELECT product_id FROM subscriptions")

    assert sql.endswith("AS guarded LIMIT 101")


def test_oracle_limit_uses_fetch_first() -> None:
    assert guard("SELECT * FROM customers", dialect="oracle").endswith("FETCH FIRST 101 ROWS ONLY")


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("DELETE FROM customers", "Only SELECT"),
        ("SELECT 1; DROP TABLE customers", "Exactly one"),
        ("SELECT * FROM pg_catalog.pg_roles", "Schema 'pg_catalog'"),
        ("SELECT * FROM invoices", "Table 'invoices'"),
        ("SELECT set_config('app.region_scope', '*', false)", "set_config"),
        ("SELECT current_setting('app.region_scope')", "current_setting"),
        ("SELECT pg_sleep(30)", "pg_sleep"),
        ("SELECT * INTO copy FROM customers", "SELECT INTO"),
        ("SELECT * FROM customers FOR UPDATE", "locking"),
    ],
)
def test_rejects_unsafe_sql(sql: str, reason: str) -> None:
    with pytest.raises(ToolError, match=reason) as info:
        guard(sql)
    assert info.value.kind == ErrorKind.INVALID_INPUT


def test_rejects_duckdb_file_access() -> None:
    with pytest.raises(ToolError, match="read_csv"):
        guard_sql(
            "SELECT * FROM read_csv('/etc/passwd')",
            dialect="duckdb",
            allowed_tables={"fact_revenue"},
            allowed_schemas={"main"},
            max_rows=10,
        )
