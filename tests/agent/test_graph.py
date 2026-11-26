import json
from pathlib import Path
from typing import Any

from analytics_platform.agent.models import ReportLinks
from analytics_platform.config import Settings
from analytics_platform.llm import FakeLLM
from tests.agent.conftest import (
    ANALYSIS,
    FINANCE,
    INTENT,
    OUTAGE_DOC,
    PLANNER,
    PRICING_DOC,
    SQL_CITATION,
    SYNTHESIS,
    WEB_SOURCE,
    FakeDocs,
    FakeSQL,
    FakeWeb,
    ask,
    doc_answer,
    draft,
    intent,
    make_toolbox,
    no_doc_answer,
    service,
)

REFUND = doc_answer(
    "Enterprise customers get a 30-day pro-rated refund; above $50k needs VP approval.",
    PRICING_DOC,
    "30-day pro-rated refund",
)


def plan(*steps: dict[str, Any], contract: list[str] | None = None) -> dict[str, Any]:
    ids = contract or ["C1"]
    return {
        "contract": [{"id": c, "description": f"item {c}"} for c in ids],
        "steps": list(steps),
    }


def step(
    sid: str, tool: str, text: str, deps: list[str] | None = None, **extra: Any
) -> dict[str, Any]:
    return {"id": sid, "tool": tool, "input": text, "depends_on": deps or [], **extra}


async def test_document_question_uses_fast_path_and_numbers_sources(
    agent_settings: Settings,
) -> None:
    llm = (
        FakeLLM()
        .when(INTENT, intent("document_qa", goal="Enterprise refund policy"))
        .when(SYNTHESIS, draft("A 30-day pro-rated refund applies [S1].", {"C1": ["S1"]}))
    )
    docs = FakeDocs(lambda q: REFUND)

    result = await ask(make_toolbox(llm, agent_settings, documents=docs), "Refund policy?")

    assert result.status == "answered" and result.answer is not None
    assert result.answer.status == "complete"
    assert result.answer.text == "A 30-day pro-rated refund applies [1]."
    assert result.answer.citations == [PRICING_DOC]
    assert docs.calls == ["Refund policy?"]
    assert not any(PLANNER in str(c.messages) for c in llm.calls)  # no planner call


async def test_data_query_with_chart_returns_table_and_chart(agent_settings: Settings) -> None:
    llm = (
        FakeLLM()
        .when(INTENT, intent("data_query", "chart", goal="Revenue by region last quarter"))
        .when(SYNTHESIS, draft("EMEA fell 28.7% [S1].", {"C1": ["S1"], "C2": ["S2"]}))
    )
    result = await ask(make_toolbox(llm, agent_settings, sql=FakeSQL()), "Chart revenue by region")

    answer = result.answer
    assert answer is not None and answer.status == "complete"
    assert [t.title for t in answer.tables] == [SQL_CITATION.title]
    assert answer.charts[0].spec.type == "grouped_bar"
    assert answer.charts[0].vega_lite["$schema"].startswith("https://vega.github.io")
    assert answer.assumptions == ["'Last quarter' is 2026 Q3, compared with 2026 Q2."]


async def test_root_cause_runs_independent_steps_in_parallel(agent_settings: Settings) -> None:
    root_plan = plan(
        step("S1", "sql", "net revenue by region last quarter vs previous", covers=["C1"]),
        step("S2", "documents", "EMEA Professional price increase", covers=["C2"]),
        step("S3", "documents", "APAC outage incident", covers=["C2"]),
        step("S4", "analysis", "each region's change in revenue", ["S1"], covers=["C1"]),
        step("S5", "chart", "Revenue change by region", ["S1"], covers=["C3"]),
        contract=["C1", "C2", "C3"],
    )
    code = {
        "code": "result = s1.assign(change=s1.net_revenue - s1.net_revenue_previous)"
        "[['region', 'change']].sort_values('change')",
        "explanation": "Revenue change per region.",
    }
    llm = (
        FakeLLM()
        .when(INTENT, intent("root_cause", goal="Explain last quarter's revenue drop"))
        .when(PLANNER, root_plan)
        .when(ANALYSIS, code)
        .when(
            SYNTHESIS,
            draft(
                "Revenue fell mainly in EMEA [S1][S4], after a price increase [S2], and in APAC "
                "after an outage [S3].",
                {"C1": ["S1", "S4"], "C2": ["S2", "S3"], "C3": ["S5"]},
            ),
        )
    )
    sql = FakeSQL()
    docs = FakeDocs(
        lambda q: (
            doc_answer("Prices rose 12%.", PRICING_DOC, "rose 12%")
            if "EMEA" in q
            else doc_answer("A 9-hour outage hit APAC.", OUTAGE_DOC, "9-hour outage")
        ),
        concurrency=sql.concurrency,
    )
    toolbox = make_toolbox(llm, agent_settings, sql=sql, documents=docs, analysis=True)

    result = await ask(toolbox, "Why did revenue drop last quarter?")

    answer = result.answer
    assert answer is not None and answer.status == "complete", answer
    assert sql.concurrency.peak == 3  # S1, S2 and S3 ran at the same time
    assert all(r.status == "done" for r in result.runs.values())
    analysis_table = next(t for t in answer.tables if t.columns == ["region", "change"])
    assert analysis_table.rows[0] == ["EMEA", -1_248_000.0]  # largest drop first
    assert answer.citations == [SQL_CITATION, PRICING_DOC, OUTAGE_DOC]
    assert answer.text.startswith("Revenue fell mainly in EMEA [1], after a price increase [2]")
    assert len(answer.charts) == 1


async def test_ambiguous_request_pauses_for_clarification_and_resumes(
    agent_settings: Settings,
) -> None:
    def classify(messages: Any) -> str:
        if "Clarification - Q:" in str(messages):
            return json.dumps(intent("data_query", goal="Net revenue by region"))
        return json.dumps(
            intent(
                "data_query",
                ambiguity="'performance' is not a defined metric",
                question="Which metric do you mean: revenue, refunds or customers?",
            )
        )

    llm = (
        FakeLLM()
        .when(INTENT, classify)
        .when(SYNTHESIS, draft("EMEA fell 28.7% [S1].", {"C1": ["S1"]}))
    )
    sql = FakeSQL()
    agent = service(make_toolbox(llm, agent_settings, sql=sql))

    first = await agent.ask("How is performance by region?", FINANCE)
    assert first.status == "needs_clarification"
    assert first.clarifying_question == "Which metric do you mean: revenue, refunds or customers?"
    assert sql.calls == []

    second = await agent.clarify(first.thread_id, "revenue")
    assert second.status == "answered" and second.answer is not None
    assert second.answer.status == "complete"
    assert "A: revenue" in sql.calls[0][0]


async def test_out_of_scope_request_is_refused_without_tools(agent_settings: Settings) -> None:
    llm = FakeLLM().when(INTENT, intent("out_of_scope"))
    sql = FakeSQL()
    result = await ask(make_toolbox(llm, agent_settings, sql=sql), "Write me a poem")
    assert result.answer is not None and result.answer.status == "refused"
    assert sql.calls == []


async def test_invalid_plan_is_sent_back_once_for_repair(agent_settings: Settings) -> None:
    bad = plan(step("S1", "chart", "Revenue chart", covers=["C1"]))
    good = plan(
        step("S1", "sql", "revenue by region", covers=["C1"]),
        step("S2", "chart", "Revenue chart", ["S1"], covers=["C1"]),
    )
    llm = (
        FakeLLM()
        .when(INTENT, intent("market_research", goal="Revenue overview"))
        .when(r"Fix these problems", good)
        .when(PLANNER, bad)
        .when(SYNTHESIS, draft("Revenue [S1].", {"C1": ["S1", "S2"]}))
    )
    result = await ask(make_toolbox(llm, agent_settings, sql=FakeSQL()), "Revenue overview")

    repair = next(c for c in llm.calls if "Fix these problems" in str(c.messages))
    assert "S1 (chart) must depend on an earlier sql or analysis step" in str(repair.messages)
    assert result.budget is not None and result.budget.replans == 1
    assert result.answer is not None and result.answer.status == "complete"


async def test_plan_still_invalid_after_repair_is_pruned_to_a_partial_answer(
    agent_settings: Settings,
) -> None:
    bad = plan(
        step("S1", "sql", "revenue by region", covers=["C1"]),
        step("S2", "web", "Competitor pricing offered to CUST-00123", covers=["C2"]),
        contract=["C1", "C2"],
    )
    llm = (
        FakeLLM()
        .when(INTENT, intent("market_research", goal="Revenue vs competitors"))
        .when(PLANNER, bad)
        .when(SYNTHESIS, draft("Revenue [S1].", {"C1": ["S1"], "C2": []}))
    )
    web = FakeWeb()
    result = await ask(
        make_toolbox(llm, agent_settings, sql=FakeSQL(), web=web), "Compare with competitors"
    )

    assert web.calls == []  # the leaking query never left the building
    answer = result.answer
    assert answer is not None and answer.status == "partial"
    assert "Could not plan how to get: item C2" in answer.gaps
    assert [s.id for s in result.plan] == ["S1"]


def _report_files(report: ReportLinks) -> tuple[str, int]:
    return Path(report.html).read_text(), Path(report.pdf).stat().st_size


async def test_market_research_report_is_rendered_and_saved(agent_settings: Settings) -> None:
    research_plan = plan(
        step("S1", "web", "Competitor promotions in APAC", queries=["CRM discount APAC 2026"],
             covers=["C1"]),
        step("S2", "sql", "net revenue by region last quarter", covers=["C2"]),
        step("S3", "chart", "Revenue by region", ["S2"], covers=["C2"]),
        contract=["C1", "C2"],
    )  # fmt: skip
    report_draft = draft(
        "A competitor discount coincided with APAC's drop [S1][S2].",
        {"C1": ["S1"], "C2": ["S2", "S3"]},
        title="Competitive pressure in APAC",
        key_findings=["A 40% competitor discount ran in APAC [S1].", "APAC fell 34% [S2]."],
        sections=[
            {"heading": "Market", "paragraphs": ["Competitor discount [S1]."], "evidence": []},
            {"heading": "Our revenue", "paragraphs": ["APAC fell [S2]."], "evidence": ["S3"]},
        ],
    )
    llm = (
        FakeLLM()
        .when(INTENT, intent("market_research", "report", goal="APAC competition report"))
        .when(PLANNER, research_plan)
        .when(SYNTHESIS, report_draft)
    )
    web = FakeWeb()
    result = await ask(
        make_toolbox(llm, agent_settings, sql=FakeSQL(), web=web), "Prepare an APAC report"
    )

    answer = result.answer
    assert answer is not None
    assert answer.status == "complete", answer.gaps
    assert web.calls == [("Competitor promotions in APAC", ["CRM discount APAC 2026"])]
    assert answer.report is not None
    html, pdf_size = _report_files(answer.report)
    assert "Competitive pressure in APAC" in html and "<svg" in html
    assert pdf_size > 1000
    assert not [w for w in answer.warnings if w.startswith("Report check")]
    assert answer.citations == [WEB_SOURCE, SQL_CITATION]


async def test_no_evidence_gives_failed_answer_without_synthesis(agent_settings: Settings) -> None:
    llm = FakeLLM().when(INTENT, intent("document_qa", goal="Holiday policy"))
    result = await ask(
        make_toolbox(llm, agent_settings, documents=FakeDocs(lambda q: no_doc_answer())),
        "What is the holiday policy?",
    )
    answer = result.answer
    assert answer is not None and answer.status == "failed"
    assert not any(SYNTHESIS in str(c.messages) for c in llm.calls)
