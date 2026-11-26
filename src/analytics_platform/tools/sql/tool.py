"""SQL tool: governed metric queries first, guarded free-form SQL as the fallback."""

import asyncio
import hashlib
import json
import logging
from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel

from analytics_platform.config import Settings
from analytics_platform.llm import BaseLLM, Message, ModelTier
from analytics_platform.security import UserContext
from analytics_platform.tools.base import Citation, ErrorKind, ToolError
from analytics_platform.tools.sql.guard import guard_sql
from analytics_platform.tools.sql.periods import (
    ResolvedPeriod,
    previous_period,
    resolve,
    same_period_last_year,
)
from analytics_platform.tools.sql.prompts import FREEFORM_SYSTEM, SEMANTIC_SYSTEM
from analytics_platform.tools.sql.semantic import MetricQuery, SemanticCompiler, SemanticModel
from analytics_platform.tools.sql.sources import (
    DataSource,
    DuckDBWarehouse,
    QueryRows,
    build_sources,
)

logger = logging.getLogger(__name__)

MAX_SQL_REPAIRS = 2


class QueryResult(BaseModel):
    method: Literal["semantic", "sql"]
    source: str
    question: str | None = None
    sql: str
    params: list[Any] = []
    columns: list[str]
    rows: list[dict[str, Any]]
    truncated: bool = False
    metric_query: MetricQuery | None = None
    period: ResolvedPeriod | None = None
    comparison_period: ResolvedPeriod | None = None
    assumptions: list[str] = []
    warnings: list[str] = []
    citation: Citation

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.columns)


class SemanticPlan(BaseModel):
    answerable: bool
    reason: str
    query: MetricQuery | None = None


class SQLDraft(BaseModel):
    source: Literal["crm", "finance", "warehouse"]
    sql: str
    explanation: str


class SQLTool:
    def __init__(
        self,
        llm: BaseLLM,
        settings: Settings,
        *,
        sources: dict[str, DataSource] | None = None,
        semantic_model: SemanticModel | None = None,
    ) -> None:
        self.llm = llm
        self.max_rows = settings.sql_max_rows
        self.timeout_s = settings.sql_timeout_s
        self.sources = sources or build_sources(settings)
        model = semantic_model or SemanticModel.load(settings.semantic_layer_path)
        self.compiler = SemanticCompiler(model)

    @property
    def warehouse(self) -> DuckDBWarehouse:
        source = self.sources["warehouse"]
        assert isinstance(source, DuckDBWarehouse)
        return source

    async def ask(self, question: str, user: UserContext) -> QueryResult:
        """Answer a data question, preferring governed metrics over free-form SQL."""
        as_of = await asyncio.to_thread(self.warehouse.as_of)
        messages: list[Message] = [
            {
                "role": "system",
                "content": SEMANTIC_SYSTEM.format(
                    as_of=as_of.isoformat(), catalog=self.compiler.model.catalog_text()
                ),
            },
            {"role": "user", "content": question},
        ]
        plan = await self.llm.structured(messages, SemanticPlan, tier=ModelTier.STRONG)
        for attempt in range(2):
            if not (plan.answerable and plan.query):
                break
            errors = self.compiler.validate(plan.query)
            if not errors:
                result = await self.run_metric_query(plan.query, user)
                result.question = question
                return result
            if attempt == 1:
                logger.info("Semantic query still invalid, falling back to SQL: %s", errors)
                break
            messages += [
                {"role": "assistant", "content": plan.model_dump_json()},
                {"role": "user", "content": "Fix these problems:\n- " + "\n- ".join(errors)},
            ]
            plan = await self.llm.structured(messages, SemanticPlan, tier=ModelTier.STRONG)
        return await self._freeform(question, user, as_of.isoformat())

    async def run_metric_query(self, q: MetricQuery, user: UserContext) -> QueryResult:
        errors = self.compiler.validate(q)
        if q.compare_to and any(self.compiler.model.dimensions[d].time_grain for d in q.dimensions):
            errors.append("compare_to cannot be combined with month, quarter or year dimensions.")
        if errors:
            raise ToolError("; ".join(errors), kind=ErrorKind.INVALID_INPUT)

        as_of = await asyncio.to_thread(self.warehouse.as_of)
        period = resolve(q.time_range, as_of) if q.time_range else None
        limit = min(q.limit or self.max_rows, self.max_rows)
        compiled = self.compiler.compile(q, period=period, user=user, limit=limit + 1)
        raw = await self._execute(self.warehouse, compiled.sql, compiled.params, user)
        rows, truncated = _to_records(raw, limit)

        assumptions = (
            [period.assumption] if period and period.assumption and not period.partial else []
        )
        warnings = [period.assumption] if period and period.partial and period.assumption else []
        if user.region_scope is not None:
            assumptions.append(
                f"Results are limited to your regions: {', '.join(user.region_scope)}."
            )

        comparison = None
        if q.compare_to and period:
            comparison = (
                previous_period(period)
                if q.compare_to == "previous_period"
                else same_period_last_year(period)
            )
            base = q.model_copy(update={"order_by": [], "limit": None})
            prev_sql = self.compiler.compile(
                base, period=comparison, user=user, limit=self.max_rows
            )
            prev_raw = await self._execute(self.warehouse, prev_sql.sql, prev_sql.params, user)
            prev_rows, _ = _to_records(prev_raw, self.max_rows)
            _merge_comparison(rows, prev_rows, q.dimensions, q.metrics)

        columns = list(rows[0]) if rows else raw.columns
        if not rows:
            warnings.append("The query returned no rows.")
        title = f"{', '.join(q.metrics)}" + (
            f" by {', '.join(q.dimensions)}" if q.dimensions else ""
        )
        if period:
            title += f" ({period.label})"
        return QueryResult(
            method="semantic",
            source=self.warehouse.name,
            sql=compiled.sql,
            params=compiled.params,
            columns=columns,
            rows=rows,
            truncated=truncated,
            metric_query=q,
            period=period,
            comparison_period=comparison,
            assumptions=assumptions,
            warnings=warnings,
            citation=_citation(f"Warehouse: {title}", compiled.sql, compiled.params),
        )

    async def run_sql(
        self, source_name: str, sql: str, user: UserContext, *, question: str | None = None
    ) -> QueryResult:
        source = self.sources.get(source_name)
        if source is None:
            raise ToolError(f"Unknown source '{source_name}'.", kind=ErrorKind.INVALID_INPUT)
        source.check_access(user)
        allowed = await asyncio.to_thread(source.allowed_tables)
        guarded = guard_sql(
            sql,
            dialect=source.dialect,
            allowed_tables=allowed,
            allowed_schemas=source.allowed_schemas,
            max_rows=self.max_rows,
        )
        raw = await self._execute(source, guarded.sql, [], user)
        rows, truncated = _to_records(raw, self.max_rows)
        warnings = [] if rows else ["The query returned no rows."]
        if truncated:
            warnings.append(f"Only the first {self.max_rows} rows are shown.")
        assumptions = []
        if user.region_scope is not None:
            assumptions.append(
                f"Results are limited to your regions: {', '.join(user.region_scope)}."
            )
        return QueryResult(
            method="sql",
            source=source.name,
            question=question,
            sql=guarded.sql,
            columns=raw.columns,
            rows=rows,
            truncated=truncated,
            assumptions=assumptions,
            warnings=warnings,
            citation=_citation(
                f"{source.name} query on {', '.join(guarded.tables)}", guarded.sql, []
            ),
        )

    async def _freeform(self, question: str, user: UserContext, as_of: str) -> QueryResult:
        usable = []
        for source in self.sources.values():
            try:
                source.check_access(user)
            except ToolError:
                continue
            usable.append(source)
        schemas = await asyncio.gather(*(asyncio.to_thread(s.schema) for s in usable))
        messages: list[Message] = [
            {
                "role": "system",
                "content": FREEFORM_SYSTEM.format(
                    as_of=as_of, schemas="\n\n".join(s.to_prompt() for s in schemas)
                ),
            },
            {"role": "user", "content": question},
        ]
        draft = await self.llm.structured(messages, SQLDraft, tier=ModelTier.STRONG)
        for attempt in range(MAX_SQL_REPAIRS + 1):
            try:
                return await self.run_sql(draft.source, draft.sql, user, question=question)
            except ToolError as exc:
                if exc.kind != ErrorKind.INVALID_INPUT or attempt == MAX_SQL_REPAIRS:
                    raise
                logger.info("Repairing SQL (attempt %d): %s", attempt + 1, exc)
                messages += [
                    {"role": "assistant", "content": draft.model_dump_json()},
                    {"role": "user", "content": f"The query failed: {exc}\nReturn a fixed query."},
                ]
                draft = await self.llm.structured(messages, SQLDraft, tier=ModelTier.STRONG)
        raise AssertionError("unreachable")

    async def _execute(
        self, source: DataSource, sql: str, params: Sequence[Any], user: UserContext
    ) -> QueryRows:
        return await asyncio.to_thread(source.execute, sql, params, user, self.timeout_s)


def _to_records(raw: QueryRows, limit: int) -> tuple[list[dict[str, Any]], bool]:
    truncated = len(raw.rows) > limit
    return [dict(zip(raw.columns, row, strict=True)) for row in raw.rows[:limit]], truncated


def _merge_comparison(
    rows: list[dict[str, Any]],
    previous: list[dict[str, Any]],
    dimensions: list[str],
    metrics: list[str],
) -> None:
    by_key = {tuple(r[d] for d in dimensions): r for r in previous}
    for row in rows:
        prev = by_key.get(tuple(row[d] for d in dimensions), {})
        for m in metrics:
            before, now = prev.get(m), row.get(m)
            row[f"{m}_previous"] = before
            row[f"{m}_change_pct"] = (
                round((now / before - 1) * 100, 2) if before and now is not None else None
            )


def _citation(title: str, sql: str, params: Sequence[Any]) -> Citation:
    digest = hashlib.sha256(json.dumps([sql, [str(p) for p in params]]).encode()).hexdigest()
    return Citation(source_type="sql", source_id=f"sql:{digest[:12]}", title=title, detail=sql)
