"""Read-only connections to the three structured data sources, with timeouts and introspection."""

import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from typing import Any

import duckdb
import oracledb
import psycopg
from pydantic import BaseModel

from analytics_platform.config import Settings
from analytics_platform.security import UserContext
from analytics_platform.tools.base import ErrorKind, ToolError

MAX_SAMPLE_VALUES = 12


class ColumnInfo(BaseModel):
    name: str
    type: str
    comment: str | None = None
    sample_values: list[str] | None = None


class TableInfo(BaseModel):
    name: str
    comment: str | None = None
    columns: list[ColumnInfo]


class SchemaInfo(BaseModel):
    source: str
    dialect: str
    description: str
    tables: list[TableInfo]

    def to_prompt(self) -> str:
        lines = [f"Source '{self.source}' ({self.dialect}): {self.description}"]
        for t in self.tables:
            lines.append(f"  TABLE {t.name}" + (f"  -- {t.comment}" if t.comment else ""))
            for c in t.columns:
                extra = []
                if c.comment:
                    extra.append(c.comment)
                if c.sample_values:
                    extra.append("values: " + ", ".join(c.sample_values))
                suffix = f"  -- {'; '.join(extra)}" if extra else ""
                lines.append(f"    {c.name} {c.type}{suffix}")
        return "\n".join(lines)


class QueryRows(BaseModel):
    columns: list[str]
    rows: list[list[Any]]


def _plain(value: Any) -> Any:
    return float(value) if isinstance(value, Decimal) else value


class DataSource(ABC):
    name: str
    dialect: str
    description: str
    allowed_schemas: frozenset[str]

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._schema: SchemaInfo | None = None

    @abstractmethod
    def _execute(
        self, sql: str, params: Sequence[Any], user: UserContext, timeout_s: float
    ) -> QueryRows: ...

    @abstractmethod
    def _introspect(self) -> list[TableInfo]: ...

    def check_access(self, user: UserContext) -> None:
        """Raise if the user may not run custom SQL here. Overridden where RLS exists."""
        if user.is_region_restricted:
            raise ToolError(
                f"Custom SQL on '{self.name}' is not available to region-restricted users. "
                "Use the governed metrics (semantic layer) instead.",
                kind=ErrorKind.PERMISSION,
            )

    def execute(
        self, sql: str, params: Sequence[Any], user: UserContext, timeout_s: float
    ) -> QueryRows:
        timeout = ToolError(f"Query exceeded {timeout_s:.0f}s", kind=ErrorKind.TIMEOUT)
        try:
            result = self._execute(sql, params, user, timeout_s)
        except psycopg.errors.QueryCanceled as exc:
            raise timeout from exc
        except psycopg.OperationalError as exc:
            raise self._unavailable(exc) from exc
        except duckdb.InterruptException as exc:
            raise timeout from exc
        except oracledb.Error as exc:
            message = str(exc)
            if "DPY-4024" in message:
                raise timeout from exc
            if isinstance(exc, oracledb.OperationalError | oracledb.InterfaceError):
                raise self._unavailable(exc) from exc
            raise self._rejected(exc, sql) from exc
        except (psycopg.Error, duckdb.Error) as exc:
            raise self._rejected(exc, sql) from exc
        result.rows = [[_plain(v) for v in row] for row in result.rows]
        return result

    def _unavailable(self, exc: Exception) -> ToolError:
        return ToolError(f"{self.name} is unavailable: {exc}", kind=ErrorKind.UNAVAILABLE)

    def _rejected(self, exc: Exception, sql: str) -> ToolError:
        return ToolError(
            f"{self.name} rejected the query: {exc}", kind=ErrorKind.INVALID_INPUT, details=sql
        )

    def schema(self) -> SchemaInfo:
        if self._schema is None:
            self._schema = SchemaInfo(
                source=self.name,
                dialect=self.dialect,
                description=self.description,
                tables=self._introspect(),
            )
        return self._schema

    def allowed_tables(self) -> set[str]:
        return {t.name.lower() for t in self.schema().tables}


class PostgresCRM(DataSource):
    name = "crm"
    dialect = "postgres"
    description = "Operational CRM: customers, subscriptions, prices, opportunities, tickets."
    allowed_schemas = frozenset({"crm"})

    def check_access(self, user: UserContext) -> None:
        """Row-level security in Postgres enforces region scope, so everyone may query."""

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self.settings.postgres_dsn(read_only=True), connect_timeout=3)

    def _execute(
        self, sql: str, params: Sequence[Any], user: UserContext, timeout_s: float
    ) -> QueryRows:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT set_config('app.region_scope', %s, true), "
                "set_config('search_path', 'crm', true), "
                "set_config('statement_timeout', %s, true)",
                (user.region_scope_setting(), str(int(timeout_s * 1000))),
            )
            cur.execute(sql, list(params))
            columns = [d.name for d in cur.description or []]
            return QueryRows(columns=columns, rows=[list(r) for r in cur.fetchall()])

    def _introspect(self) -> list[TableInfo]:
        sql = """
            SELECT c.table_name, c.column_name, c.data_type,
                   col_description(format('crm.%I', c.table_name)::regclass, c.ordinal_position),
                   obj_description(format('crm.%I', c.table_name)::regclass, 'pg_class')
            FROM information_schema.columns c
            WHERE c.table_schema = 'crm'
            ORDER BY c.table_name, c.ordinal_position
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT set_config('app.region_scope', '*', true)")
            cur.execute(sql)
            raw = cur.fetchall()
            tables = _group_columns(raw)
            for table in tables:
                for col in table.columns:
                    if col.type in ("text", "character"):
                        cur.execute(
                            f'SELECT DISTINCT "{col.name}"::text FROM crm."{table.name}" '
                            f'WHERE "{col.name}" IS NOT NULL LIMIT {MAX_SAMPLE_VALUES + 1}'
                        )
                        col.sample_values = _samples(cur.fetchall())
        return tables


class OracleFinance(DataSource):
    name = "finance"
    dialect = "oracle"
    description = "Finance system: invoices, refunds and SLA credits, monthly budgets."

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.owner = settings.oracle_user.upper()
        self.allowed_schemas = frozenset({settings.oracle_user.lower()})

    def _connect(self) -> oracledb.Connection:
        conn = oracledb.connect(
            user=self.settings.oracle_ro_user,
            password=self.settings.oracle_ro_password.get_secret_value(),
            dsn=self.settings.oracle_dsn,
            tcp_connect_timeout=3,
        )
        conn.current_schema = self.owner
        return conn

    def _execute(
        self, sql: str, params: Sequence[Any], user: UserContext, timeout_s: float
    ) -> QueryRows:
        with self._connect() as conn:
            conn.call_timeout = int(timeout_s * 1000)
            cur = conn.cursor()
            cur.execute(sql, list(params))
            columns = [d[0].lower() for d in cur.description or []]
            return QueryRows(columns=columns, rows=[list(r) for r in cur.fetchall()])

    def _introspect(self) -> list[TableInfo]:
        sql = """
            SELECT c.table_name, c.column_name, c.data_type, cc.comments, tc.comments
            FROM all_tab_columns c
            LEFT JOIN all_col_comments cc
              ON cc.owner = c.owner AND cc.table_name = c.table_name
             AND cc.column_name = c.column_name
            LEFT JOIN all_tab_comments tc ON tc.owner = c.owner AND tc.table_name = c.table_name
            WHERE c.owner = :owner
            ORDER BY c.table_name, c.column_id
        """
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(sql, owner=self.owner)
            raw = [(t.lower(), c.lower(), dt.lower(), cc, tc) for t, c, dt, cc, tc in cur]
            tables = _group_columns(raw)
            for table in tables:
                for col in table.columns:
                    if col.type.startswith("varchar") and not col.name.endswith("_id"):
                        cur.execute(
                            f"SELECT DISTINCT {col.name} FROM {table.name} "
                            f"WHERE {col.name} IS NOT NULL FETCH FIRST {MAX_SAMPLE_VALUES + 1} "
                            "ROWS ONLY"
                        )
                        col.sample_values = _samples(cur.fetchall())
        return tables


class DuckDBWarehouse(DataSource):
    name = "warehouse"
    dialect = "duckdb"
    description = (
        "Analytics warehouse (star schema). Net revenue = SUM(fact_revenue.amount_usd); "
        "refund rows are negative."
    )
    allowed_schemas = frozenset({"main"})

    def _connect(self) -> duckdb.DuckDBPyConnection:
        try:
            return duckdb.connect(
                self.settings.duckdb_path,
                read_only=True,
                config={"enable_external_access": False},
            )
        except duckdb.IOException as exc:
            raise ToolError(f"Warehouse is unavailable: {exc}", kind=ErrorKind.UNAVAILABLE) from exc

    def _execute(
        self, sql: str, params: Sequence[Any], user: UserContext, timeout_s: float
    ) -> QueryRows:
        con = self._connect()
        timer = threading.Timer(timeout_s, con.interrupt)
        timer.start()
        try:
            cur = con.execute(sql, list(params))
            columns = [d[0] for d in cur.description or []]
            return QueryRows(columns=columns, rows=[list(r) for r in cur.fetchall()])
        finally:
            timer.cancel()
            con.close()

    def as_of(self) -> date:
        con = self._connect()
        try:
            row = con.execute("SELECT data_as_of FROM warehouse_info").fetchone()
        finally:
            con.close()
        assert row is not None
        value: date = row[0]
        return value

    def _introspect(self) -> list[TableInfo]:
        con = self._connect()
        try:
            raw = con.execute(
                """
                SELECT c.table_name, c.column_name, lower(c.data_type), c.comment, t.comment
                FROM duckdb_columns() c JOIN duckdb_tables() t USING (table_name)
                WHERE c.schema_name = 'main' AND c.table_name <> 'warehouse_info'
                ORDER BY c.table_name, c.column_index
                """
            ).fetchall()
            tables = _group_columns(raw)
            for table in tables:
                for col in table.columns:
                    if col.type == "varchar" and not col.name.endswith("_id"):
                        rows = con.execute(
                            f'SELECT DISTINCT "{col.name}" FROM "{table.name}" '
                            f'WHERE "{col.name}" IS NOT NULL LIMIT {MAX_SAMPLE_VALUES + 1}'
                        ).fetchall()
                        col.sample_values = _samples(rows)
        finally:
            con.close()
        return tables


def _group_columns(raw: Sequence[Sequence[Any]]) -> list[TableInfo]:
    tables: dict[str, TableInfo] = {}
    for table, column, dtype, column_comment, table_comment in raw:
        info = tables.setdefault(table, TableInfo(name=table, comment=table_comment, columns=[]))
        info.columns.append(ColumnInfo(name=column, type=dtype, comment=column_comment))
    return list(tables.values())


def _samples(rows: Sequence[Sequence[Any]]) -> list[str] | None:
    """Distinct values for low-cardinality columns only; high-cardinality columns get None."""
    if len(rows) > MAX_SAMPLE_VALUES:
        return None
    return sorted(str(r[0]) for r in rows)


def build_sources(settings: Settings) -> dict[str, DataSource]:
    sources: list[DataSource] = [
        PostgresCRM(settings),
        OracleFinance(settings),
        DuckDBWarehouse(settings),
    ]
    return {s.name: s for s in sources}
