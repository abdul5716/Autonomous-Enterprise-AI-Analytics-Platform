from datetime import date

import pytest

from analytics_platform.agent.checkpoint import serializer, state_types
from analytics_platform.agent.models import (
    Budget,
    ChartData,
    Evidence,
    Intent,
    Plan,
    PlanStep,
    StepRun,
    TableData,
)
from analytics_platform.agent.planning import check_plan, ensure_chart, prune
from analytics_platform.agent.render import Numbering
from analytics_platform.agent.steps import jsonable
from analytics_platform.config import Settings
from analytics_platform.llm import FakeLLM
from analytics_platform.reliability import CircuitBreaker, CircuitOpenError, retry_async
from analytics_platform.tools.base import Citation, ErrorKind, ToolError
from analytics_platform.tools.charts import ChartSpec
from tests.agent.conftest import (
    INTENT,
    PRICING_DOC,
    SQL_CITATION,
    SYNTHESIS,
    FakeDocs,
    FakeSQL,
    ask,
    draft,
    intent,
    make_toolbox,
    revenue_result,
)

DATA_QUERY = intent("data_query", goal="Revenue by region")
ANSWER = draft("EMEA fell 28.7% [S1].", {"C1": ["S1"]})


def llm_for(kind: dict[str, object] = DATA_QUERY) -> FakeLLM:
    return FakeLLM().when(INTENT, kind).when(SYNTHESIS, ANSWER)


async def test_transient_error_is_retried(agent_settings: Settings) -> None:
    failures = iter([ToolError("connection reset", kind=ErrorKind.TRANSIENT)])

    def flaky(q: str, excluded: frozenset[str]):  # type: ignore[no-untyped-def]
        error = next(failures, None)
        if error:
            raise error
        return revenue_result(q)

    result = await ask(make_toolbox(llm_for(), agent_settings, sql=FakeSQL(flaky)), "Revenue?")

    assert result.runs["S1"].status == "done" and result.runs["S1"].attempts == 2
    assert result.budget is not None and result.budget.tool_calls == 2
    assert result.answer is not None and result.answer.status == "complete"


async def test_finance_database_down_falls_back_to_other_sources(agent_settings: Settings) -> None:
    def oracle_down(q: str, excluded: frozenset[str]):  # type: ignore[no-untyped-def]
        if "finance" not in excluded:
            raise ToolError("finance is unavailable: DPY-6005", kind=ErrorKind.UNAVAILABLE,
                            source="finance")  # fmt: skip
        return revenue_result(q)

    sql = FakeSQL(oracle_down)
    result = await ask(make_toolbox(llm_for(), agent_settings, sql=sql), "Refunds by region?")

    assert [excluded for _, excluded in sql.calls] == [frozenset(), frozenset({"finance"})]
    assert result.runs["S1"].fallback == "answered without the finance database"
    answer = result.answer
    assert answer is not None and answer.status == "complete"
    assert any("finance database was unavailable" in w for w in answer.warnings)


async def test_open_circuit_skips_a_dependency_that_keeps_failing(agent_settings: Settings) -> None:
    def down(q: str) -> object:
        raise ToolError("Document store is unavailable", kind=ErrorKind.UNAVAILABLE)

    docs = FakeDocs(down)  # type: ignore[arg-type]
    breaker = CircuitBreaker(threshold=1, cooldown_s=60)
    llm = FakeLLM().when(INTENT, intent("document_qa"))
    toolbox = make_toolbox(llm, agent_settings, documents=docs, breaker=breaker)

    first = await ask(toolbox, "Refund policy?")
    assert first.runs["S1"].attempts == 1  # unavailable is not retried
    assert breaker.state("documents") == "open"

    second = await ask(toolbox, "Refund policy?")
    assert len(docs.calls) == 1  # the open circuit stopped the second call
    assert "temporarily disabled" in (second.runs["S1"].error or "")
    assert second.answer is not None and second.answer.status == "failed"


async def test_exhausted_budget_skips_remaining_steps_with_a_gap(agent_settings: Settings) -> None:
    settings = agent_settings.model_copy(update={"agent_max_tool_calls": 1})
    llm = (
        FakeLLM()
        .when(INTENT, intent("data_query", "chart", goal="Revenue chart"))
        .when(SYNTHESIS, draft("EMEA fell [S1].", {"C1": ["S1"], "C2": ["S2"]}))
    )
    result = await ask(make_toolbox(llm, settings, sql=FakeSQL()), "Chart revenue")

    assert result.runs["S2"].status == "skipped"
    answer = result.answer
    assert answer is not None and answer.status == "partial"
    assert any("tool call budget (1) used up" in g for g in answer.gaps)
    assert answer.tables and not answer.charts


async def test_permission_error_fails_fast_and_skips_dependents(agent_settings: Settings) -> None:
    def denied(q: str, excluded: frozenset[str]):  # type: ignore[no-untyped-def]
        raise ToolError("You can't query the finance database.", kind=ErrorKind.PERMISSION)

    llm = FakeLLM().when(INTENT, intent("data_query", "chart"))
    sql = FakeSQL(denied)
    result = await ask(make_toolbox(llm, agent_settings, sql=sql), "Chart refunds")

    assert len(sql.calls) == 1
    assert result.runs["S1"].error_kind == "permission"
    assert result.runs["S2"].status == "skipped"
    answer = result.answer
    assert answer is not None and answer.status == "failed"
    assert any("can't query the finance database" in g for g in answer.gaps)


async def test_retry_async_backs_off_then_gives_up() -> None:
    calls = 0

    async def always_times_out() -> None:
        nonlocal calls
        calls += 1
        raise TimeoutError

    with pytest.raises(ToolError) as info:
        await retry_async(always_times_out, attempts=3, base_delay_s=0)
    assert calls == 3 and info.value.kind == ErrorKind.TIMEOUT


def test_circuit_breaker_half_opens_after_cooldown() -> None:
    now = [0.0]
    breaker = CircuitBreaker(threshold=2, cooldown_s=10, clock=lambda: now[0])
    breaker.record_failure("finance")
    breaker.check("finance")  # one failure: still closed
    breaker.record_failure("finance")
    with pytest.raises(CircuitOpenError):
        breaker.check("finance")
    now[0] = 11
    assert breaker.state("finance") == "half-open" and not breaker.is_open("finance")
    breaker.check("finance")  # the single probe is let through
    assert breaker.is_open("finance")
    with pytest.raises(CircuitOpenError):
        breaker.check("finance")  # ... and nothing else until it reports back
    breaker.record_success("finance")
    assert breaker.state("finance") == "closed"


def _plan(*steps: PlanStep, contract: tuple[str, ...] = ("C1",)) -> Plan:
    from analytics_platform.agent.models import ContractItem

    return Plan(
        contract=[ContractItem(id=c, description=f"item {c}") for c in contract], steps=list(steps)
    )


def test_check_plan_reports_structural_problems() -> None:
    plan = _plan(
        PlanStep(id="S1", tool="sql", input="revenue", covers=["C1"]),
        PlanStep(id="S2", tool="chart", input="chart", depends_on=["S3"]),
        PlanStep(id="S3", tool="web", input="Pricing in MEMO-PRC-014", covers=["C9"]),
        contract=("C1", "C2"),
    )
    problems = check_plan(plan, {"sql", "chart"}, max_steps=8)

    assert problems["S2"] == [
        "S2 depends on S3, which is not an earlier step.",
        "S2 (chart) must depend on an earlier sql or analysis step.",
    ]
    assert "S3 uses the web tool, which is not available now." in problems["S3"]
    assert (
        "S3 web input 'Pricing in MEMO-PRC-014' contains internal document id." in (problems["S3"])
    )
    assert "S3 covers unknown contract items ['C9']." in problems["S3"]
    assert problems[""] == ["Contract item C2 (item C2) is not covered by any step."]

    kept, gaps = prune(plan, problems)
    assert [s.id for s in kept] == ["S1"]
    assert gaps == ["Could not plan how to get: item C2"]


def test_ensure_chart_adds_a_missing_chart_step() -> None:
    plan = _plan(PlanStep(id="S1", tool="sql", input="revenue", covers=["C1"]))
    wants_chart = Intent(request_type="root_cause", output_format="chart", goal="g")
    fixed = ensure_chart(plan, wants_chart)
    assert fixed.steps[-1] == PlanStep(id="S2", tool="chart", input="g", depends_on=["S1"])
    assert check_plan(fixed, {"sql", "chart"}, max_steps=8) == {}


def test_numbering_assigns_numbers_in_order_of_first_use() -> None:
    web = [
        Citation(source_type="web", source_id="W1", title="a"),
        Citation(source_type="web", source_id="W2", title="b"),
    ]
    evidence = {
        "S1": Evidence(step_id="S1", tool="sql", found=True, summary="", citations=[SQL_CITATION]),
        "S2": Evidence(step_id="S2", tool="web", found=True, summary="", citations=web),
        "S3": Evidence(step_id="S3", tool="documents", found=False, summary=""),
    }
    numbering = Numbering(evidence)
    assert numbering.apply("Web says [S2]. Data [S1, S2].") == "Web says [1][2]. Data [3][1][2]."
    assert numbering.apply("Both [S1][S2][S1].") == "Both [3][1][2]."
    assert numbering.apply("Missing [S3] and [S9].") == "Missing and."
    assert numbering.citations == [*web, SQL_CITATION]
    assert numbering.unknown == {"S3", "S9"}


def test_budget_reports_what_ran_out() -> None:
    budget = Budget(max_tool_calls=5, max_llm_calls=10, max_cost_usd=0.1, max_seconds=60,
                    max_replans=1)  # fmt: skip
    assert budget.exhausted() is None
    assert budget.model_copy(update={"cost_usd": 0.1}).exhausted() == "cost budget ($0.10) used up"
    assert budget.model_copy(update={"started_at": 0}).exhausted() == "time budget (60s) used up"


def test_checkpoint_serializer_round_trips_state_and_allows_only_known_types() -> None:
    names = {t.__name__ for t in state_types()}
    assert {"Evidence", "Citation", "ChartSpec", "UserContext", "TraceEvent", "Answer"} <= names

    chart = ChartData(
        spec=ChartSpec(type="bar", title="t", x="region", y=["net_revenue"]),
        vega_lite={"mark": "bar"},
        data=TableData(columns=["region"], rows=[["EMEA"]], title="t"),
    )
    state = {
        "runs": {"S1": StepRun(step_id="S1", status="done")},
        "evidence": {
            "S1": Evidence(
                step_id="S1",
                tool="chart",
                found=True,
                summary="s",
                chart=chart,
                citations=[PRICING_DOC],
            )
        },
    }
    serde = serializer()
    assert serde.loads_typed(serde.dumps_typed(state)) == state


def test_jsonable_converts_database_values() -> None:
    from decimal import Decimal

    assert jsonable({"d": date(2026, 7, 1), "n": Decimal("1.5"), "l": (1, 2)}) == {
        "d": "2026-07-01",
        "n": 1.5,
        "l": [1, 2],
    }
