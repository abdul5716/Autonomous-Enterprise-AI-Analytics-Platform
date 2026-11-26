"""Static checks on LLM-written SQL before it reaches a database."""

from collections.abc import Set
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from analytics_platform.tools.base import ErrorKind, ToolError

# Functions that read files, sleep, touch other sessions, or change session settings (which could
# switch off row-level security). Prefixes match whole function families.
BLOCKED_FUNCTIONS = {
    "postgres": (
        "pg_sleep",
        "pg_read",
        "pg_ls_",
        "pg_stat_file",
        "pg_terminate",
        "pg_cancel",
        "lo_",
        "dblink",
        "set_config",
        "current_setting",
        "query_to_xml",
    ),
    "oracle": ("dbms_", "utl_", "sys_context", "httpuritype"),
    "duckdb": ("read_", "glob", "parquet_", "sniff_csv", "getenv", "current_setting", "query"),
}


@dataclass(frozen=True)
class GuardedSQL:
    sql: str
    tables: list[str]
    limit: int


def _reject(message: str, sql: str) -> ToolError:
    return ToolError(message, kind=ErrorKind.INVALID_INPUT, details=sql)


def guard_sql(
    sql: str,
    *,
    dialect: str,
    allowed_tables: Set[str],
    allowed_schemas: Set[str],
    max_rows: int,
) -> GuardedSQL:
    """Allow one read-only query over allowed tables; enforce a row limit of `max_rows`.

    The returned SQL fetches `max_rows + 1` rows so the caller can detect truncation.
    """
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except ParseError as exc:
        raise _reject(f"SQL does not parse: {exc}", sql) from exc
    if len(statements) != 1:
        raise _reject("Exactly one SQL statement is allowed.", sql)
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise _reject(f"Only SELECT queries are allowed, got {tree.key.upper()}.", sql)
    if tree.find(exp.Into, exp.Lock) is not None:
        raise _reject("SELECT INTO and locking clauses are not allowed.", sql)

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    tables = []
    for table in tree.find_all(exp.Table):
        name, schema = table.name.lower(), table.db.lower()
        if not name or (not schema and name in cte_names):
            continue
        if schema and schema not in allowed_schemas:
            raise _reject(f"Schema '{schema}' is not accessible.", sql)
        if name not in allowed_tables:
            raise _reject(
                f"Table '{name}' is not accessible. Allowed: {', '.join(sorted(allowed_tables))}.",
                sql,
            )
        tables.append(name)

    blocked = BLOCKED_FUNCTIONS.get(dialect, ())
    for func in tree.find_all(exp.Func):
        fname = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()
        if fname.startswith(blocked):
            raise _reject(f"Function '{fname}' is not allowed.", sql)

    limited = _apply_limit(tree, max_rows + 1)
    return GuardedSQL(sql=limited.sql(dialect=dialect), tables=sorted(set(tables)), limit=max_rows)


def _apply_limit(tree: exp.Query, cap: int) -> exp.Query:
    if isinstance(tree, exp.Select):
        existing = tree.args.get("limit") or tree.args.get("fetch")
        if existing is None:
            return tree.limit(cap)
        count = existing.args.get("expression") or existing.args.get("count")
        if isinstance(count, exp.Literal) and count.is_int and int(count.this) <= cap:
            return tree
        return tree.limit(cap)
    return exp.select("*").from_(tree.subquery("guarded")).limit(cap)
