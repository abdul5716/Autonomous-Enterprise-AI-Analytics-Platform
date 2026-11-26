"""Running one plan step: tool adapters, retries, circuit breaking and source fallback."""

import json
import logging
import time
from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from analytics_platform.agent.models import ChartData, Evidence, PlanStep, StepRun, TableData
from analytics_platform.agent.toolbox import Toolbox
from analytics_platform.llm import Usage, track_usage
from analytics_platform.reliability import CircuitOpenError, as_tool_error, retry_async
from analytics_platform.security import UserContext
from analytics_platform.tools.base import Citation, ErrorKind, ToolError
from analytics_platform.tools.charts import suggest_chart, to_vega_lite

logger = logging.getLogger(__name__)

SUMMARY_ROWS = 25
SQL_SOURCES = ("crm", "finance", "warehouse")
DEPENDENCY_FAILURES = (ErrorKind.TRANSIENT, ErrorKind.TIMEOUT, ErrorKind.UNAVAILABLE)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def table_frame(table: TableData) -> pd.DataFrame:
    return pd.DataFrame(table.rows, columns=table.columns)


def table_text(table: TableData, limit: int = SUMMARY_ROWS) -> str:
    def cell(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.2f}"
        return "" if v is None else str(v)

    lines = [" | ".join(table.columns)]
    lines += [" | ".join(cell(v) for v in row) for row in table.rows[:limit]]
    if len(table.rows) > limit:
        lines.append(f"... {len(table.rows) - limit} more rows")
    return "\n".join(lines)


def unique_citations(groups: Sequence[Sequence[Citation]]) -> list[Citation]:
    seen: dict[tuple[str, str], Citation] = {}
    for group in groups:
        for c in group:
            seen.setdefault((c.source_type, c.source_id), c)
    return list(seen.values())


async def execute_step(
    step: PlanStep, deps: dict[str, Evidence], user: UserContext, toolbox: Toolbox
) -> tuple[Evidence | None, StepRun, Usage]:
    """Run a step to completion or classified failure. Never raises for tool errors."""
    run = StepRun(step_id=step.id)
    started = time.monotonic()
    breaker = toolbox.breaker
    excluded = {s for s in SQL_SOURCES if breaker.is_open(f"sql:{s}")}
    fallback_notes: list[str] = []

    async def call() -> Evidence:
        run.attempts += 1
        return await _invoke(step, deps, user, toolbox, frozenset(excluded))

    with track_usage() as usage:
        while True:
            try:
                breaker.check(step.tool)
                evidence = await retry_async(
                    call,
                    attempts=toolbox.settings.agent_retry_attempts,
                    base_delay_s=toolbox.settings.agent_retry_base_delay_s,
                    on_retry=lambda n, e: logger.info("Retrying %s (%d): %s", step.id, n, e),
                )
            except ToolError as exc:
                if isinstance(exc, CircuitOpenError):
                    run.status, run.error, run.error_kind = "failed", str(exc), exc.kind.value
                    break
                key = _breaker_key(step, exc)
                if exc.kind in DEPENDENCY_FAILURES:
                    breaker.record_failure(key)
                if _can_fall_back(step, exc, excluded):
                    assert exc.source is not None
                    excluded.add(exc.source)
                    run.fallback = f"answered without the {exc.source} database"
                    fallback_notes.append(
                        f"The {exc.source} database was unavailable, so this was answered "
                        "from the other sources."
                    )
                    continue
                run.status, run.error, run.error_kind = "failed", str(exc), exc.kind.value
                break
            breaker.record_success(step.tool)
            evidence.warnings = fallback_notes + evidence.warnings
            run.status = "done"
            run.duration_s = round(time.monotonic() - started, 2)
            return evidence, run, usage
    run.duration_s = round(time.monotonic() - started, 2)
    return None, run, usage


def _breaker_key(step: PlanStep, exc: ToolError) -> str:
    if step.tool == "sql" and exc.source in SQL_SOURCES:
        return f"sql:{exc.source}"
    return step.tool


def _can_fall_back(step: PlanStep, exc: ToolError, excluded: set[str]) -> bool:
    """Free-form SQL can avoid a database that is down; the warehouse has no substitute."""
    return (
        step.tool == "sql"
        and exc.kind in DEPENDENCY_FAILURES
        and exc.source in ("crm", "finance")
        and exc.source not in excluded
    )


async def _invoke(
    step: PlanStep,
    deps: dict[str, Evidence],
    user: UserContext,
    toolbox: Toolbox,
    excluded: frozenset[str],
) -> Evidence:
    try:
        match step.tool:
            case "sql":
                return await _sql(step, user, toolbox, excluded)
            case "documents":
                return await _documents(step, user, toolbox)
            case "web":
                return await _web(step, user, toolbox)
            case "analysis":
                return await _analysis(step, deps, user, toolbox)
            case "chart":
                return _chart(step, deps)
    except ToolError:
        raise
    except Exception as exc:
        raise as_tool_error(exc) from exc


def _require[T](tool: T | None, name: str) -> T:
    if tool is None:
        raise ToolError(f"The {name} tool is not available.", kind=ErrorKind.UNAVAILABLE)
    return tool


async def _sql(
    step: PlanStep, user: UserContext, toolbox: Toolbox, excluded: frozenset[str]
) -> Evidence:
    result = await _require(toolbox.sql, "sql").ask(step.input, user, exclude_sources=excluded)
    table = TableData(
        columns=result.columns,
        rows=[[jsonable(row.get(c)) for c in result.columns] for row in result.rows],
        title=result.citation.title,
        truncated=result.truncated,
    )
    lines = [f"{result.citation.title} (from {result.source}, {result.method} query)"]
    lines += [f"Assumption: {a}" for a in result.assumptions]
    lines.append(table_text(table) if result.rows else "No rows.")
    return Evidence(
        step_id=step.id,
        tool="sql",
        found=bool(result.rows),
        summary="\n".join(lines),
        table=table,
        citations=[result.citation],
        assumptions=result.assumptions,
        warnings=result.warnings,
        detail=jsonable(
            {
                "source": result.source,
                "method": result.method,
                "sql": result.sql,
                "params": result.params,
            }
        ),
    )


async def _documents(step: PlanStep, user: UserContext, toolbox: Toolbox) -> Evidence:
    answer = await _require(toolbox.documents, "documents").ask(step.input, user)
    if answer.answered:
        lines = [answer.answer or ""]
        lines += [f'- {c.statement} (quote: "{c.quote}")' for c in answer.claims]
        sources = "; ".join(
            c.title + (f", {c.detail}" if c.detail else "") for c in answer.citations
        )
        lines.append(f"Documents: {sources}")
        summary = "\n".join(lines)
    else:
        summary = f"The documents you can access do not answer this: {answer.reason}"
    return Evidence(
        step_id=step.id,
        tool="documents",
        found=answer.answered,
        summary=summary,
        citations=answer.citations,
        warnings=answer.warnings,
        detail=jsonable(
            {
                "evidence": answer.retrieval.evidence,
                "claims": [c.model_dump() for c in answer.claims],
            }
        ),
    )


async def _web(step: PlanStep, user: UserContext, toolbox: Toolbox) -> Evidence:
    research = await _require(toolbox.web, "web").research(
        step.input, user, queries=step.queries or None
    )
    if research.answered:
        lines = [research.summary or ""]
        lines += [f'- {f.statement} (quote: "{f.quote}")' for f in research.findings]
        lines += [f"Conflict: {c}" for c in research.conflicts]
        lines += [
            f"{s.source_id}: {s.page.title} ({s.kind}, {s.page.domain}, "
            f"published {s.page.published or 'unknown'})"
            for s in research.sources
        ]
        summary = "\n".join(lines)
    else:
        summary = f"Web research found nothing usable: {research.reason}"
    return Evidence(
        step_id=step.id,
        tool="web",
        found=research.answered,
        summary=summary,
        citations=research.citations,
        warnings=research.warnings,
        detail=jsonable(
            {
                "queries": research.queries,
                "providers": research.providers_used,
                "findings": [f.model_dump() for f in research.findings],
            }
        ),
    )


def _tables(deps: dict[str, Evidence]) -> dict[str, TableData]:
    return {sid: ev.table for sid, ev in deps.items() if ev.table is not None and ev.table.rows}


async def _analysis(
    step: PlanStep, deps: dict[str, Evidence], user: UserContext, toolbox: Toolbox
) -> Evidence:
    tables = _tables(deps)
    if not tables:
        raise ToolError(
            "Analysis needs a table from an earlier step.", kind=ErrorKind.INVALID_INPUT
        )
    datasets = {sid.lower(): table_frame(t) for sid, t in tables.items()}
    result = await _require(toolbox.analysis, "analysis").analyze(step.input, datasets, user)
    frame = result.to_frame()
    table = None
    lines = [result.explanation]
    if frame is not None:
        table = TableData(
            columns=[str(c) for c in frame.columns],
            rows=jsonable(frame.astype(object).where(frame.notna(), None).values.tolist()),
            title=step.input,
        )
        lines.append(table_text(table))
    elif result.output.result is not None:
        lines.append("Result: " + json.dumps(result.output.result.get("value"), default=str))
    return Evidence(
        step_id=step.id,
        tool="analysis",
        found=True,
        summary="\n".join(lines),
        table=table,
        citations=unique_citations([deps[sid].citations for sid in tables]),
        detail=jsonable(
            {
                "code": result.code,
                "attempts": result.attempts,
                "inputs": result.inputs,
                "result": result.output.result,
                "stdout": result.output.stdout[-2000:],
            }
        ),
    )


def _chart(step: PlanStep, deps: dict[str, Evidence]) -> Evidence:
    tables = _tables(deps)
    if not tables:
        raise ToolError("A chart needs a table from an earlier step.", kind=ErrorKind.INVALID_INPUT)
    source_id, table = list(tables.items())[-1]
    frame = table_frame(table)
    spec = suggest_chart(frame, title=step.input or table.title)
    if spec is None:
        return Evidence(
            step_id=step.id,
            tool="chart",
            found=False,
            summary=f"The data from {source_id} is better shown as a table than a chart.",
        )
    return Evidence(
        step_id=step.id,
        tool="chart",
        found=True,
        summary=f"{spec.type.replace('_', ' ')} chart of {', '.join(spec.y)} by {spec.x}, "
        f"from {source_id}.",
        chart=ChartData(spec=spec, vega_lite=to_vega_lite(spec, frame), data=table),
        citations=deps[source_id].citations,
    )
