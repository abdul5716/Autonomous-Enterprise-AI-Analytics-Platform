import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from analytics_platform.agent.checkpoint import memory_checkpointer
from analytics_platform.agent.graph import build_graph
from analytics_platform.agent.service import AgentService, RunResult
from analytics_platform.agent.toolbox import Toolbox
from analytics_platform.config import Settings
from analytics_platform.llm import FakeLLM
from analytics_platform.reliability import CircuitBreaker
from analytics_platform.security import DEMO_USERS, UserContext
from analytics_platform.tools.base import Citation
from analytics_platform.tools.rag.tool import Claim, DocumentTool
from analytics_platform.tools.sandbox.tool import AnalysisTool
from analytics_platform.tools.sql.tool import QueryResult, SQLTool
from analytics_platform.tools.web.tool import Finding, WebResearchTool

INTENT = r"You classify requests"
PLANNER = r"You plan how to answer"
SYNTHESIS = r"You write the final answer"
ANALYSIS = r"Write Python \(pandas/numpy\) code"

FINANCE = DEMO_USERS["finance.admin"]

SQL_CITATION = Citation(
    source_type="sql",
    source_id="sql:rev",
    title="Warehouse: net_revenue by region (2026 Q3)",
    detail="SELECT region, SUM(net_revenue) ...",
)
PRICING_DOC = Citation(
    source_type="document", source_id="DOC-PRC-014", title="EMEA pricing change memo"
)
OUTAGE_DOC = Citation(
    source_type="document", source_id="DOC-INC-031", title="APAC outage postmortem"
)
WEB_SOURCE = Citation(
    source_type="web",
    source_id="W1",
    title="Competitor launches promotion",
    url="https://news.example/a",
)


def intent(
    request_type: str,
    output_format: str = "text",
    goal: str = "Answer the question",
    *,
    ambiguity: str | None = None,
    question: str | None = None,
) -> dict[str, Any]:
    return {
        "request_type": request_type,
        "output_format": output_format,
        "goal": goal,
        "ambiguity": ambiguity,
        "clarifying_question": question,
    }


def draft(answer: str, coverage: dict[str, list[str]], **extra: Any) -> dict[str, Any]:
    return {
        "answer": answer,
        "coverage": [
            {"item_id": item, "addressed": bool(ev), "evidence": ev}
            for item, ev in coverage.items()
        ],
        "confidence": "high",
        **extra,
    }


def revenue_result(question: str = "net revenue by region") -> QueryResult:
    rows = [
        ("EMEA", 3_100_000.0, 4_348_000.0, -28.71),
        ("APAC", 1_200_000.0, 1_818_000.0, -33.99),
        ("NA", 5_000_000.0, 5_200_000.0, -3.85),
        ("LATAM", 800_000.0, 790_000.0, 1.27),
    ]
    columns = ["region", "net_revenue", "net_revenue_previous", "net_revenue_change_pct"]
    return QueryResult(
        method="semantic",
        source="warehouse",
        question=question,
        sql="SELECT region, SUM(net_revenue) ...",
        columns=columns,
        rows=[dict(zip(columns, r, strict=True)) for r in rows],
        assumptions=["'Last quarter' is 2026 Q3, compared with 2026 Q2."],
        citation=SQL_CITATION,
    )


@dataclass
class Concurrency:
    active: int = 0
    peak: int = 0

    async def hold(self) -> None:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1


SqlHandler = Callable[[str, frozenset[str]], QueryResult]


@dataclass
class FakeSQL:
    handler: SqlHandler = lambda q, excluded: revenue_result(q)
    concurrency: Concurrency = field(default_factory=Concurrency)
    calls: list[tuple[str, frozenset[str]]] = field(default_factory=list)
    compiler: Any = field(
        default_factory=lambda: SimpleNamespace(
            model=SimpleNamespace(catalog_text=lambda: "Metrics:\n- net_revenue")
        )
    )

    async def ask(
        self, question: str, user: UserContext, *, exclude_sources: frozenset[str] = frozenset()
    ) -> QueryResult:
        self.calls.append((question, frozenset(exclude_sources)))
        await self.concurrency.hold()
        return self.handler(question, frozenset(exclude_sources))


def doc_answer(text: str, citation: Citation, quote: str) -> SimpleNamespace:
    return SimpleNamespace(
        answered=True,
        answer=text,
        claims=[Claim(statement=text, chunk_ids=[f"{citation.source_id}@1#0"], quote=quote)],
        citations=[citation],
        warnings=[],
        reason=None,
        retrieval=SimpleNamespace(evidence="strong"),
    )


def no_doc_answer(reason: str = "No relevant passages.") -> SimpleNamespace:
    return SimpleNamespace(
        answered=False,
        answer=None,
        claims=[],
        citations=[],
        warnings=[],
        reason=reason,
        retrieval=SimpleNamespace(evidence="none"),
    )


@dataclass
class FakeDocs:
    handler: Callable[[str], SimpleNamespace]
    concurrency: Concurrency = field(default_factory=Concurrency)
    calls: list[str] = field(default_factory=list)

    async def ask(self, question: str, user: UserContext) -> SimpleNamespace:
        self.calls.append(question)
        await self.concurrency.hold()
        return self.handler(question)


@dataclass
class FakeWeb:
    calls: list[tuple[str, list[str] | None]] = field(default_factory=list)

    async def research(
        self, question: str, user: UserContext, *, queries: list[str] | None = None
    ) -> SimpleNamespace:
        self.calls.append((question, queries))
        page = SimpleNamespace(
            title=WEB_SOURCE.title, domain="news.example", published="2026-08-01"
        )
        return SimpleNamespace(
            answered=True,
            summary="A competitor ran a 40% discount in APAC.",
            findings=[
                Finding(statement="40% APAC discount", source_ids=["W1"], quote="40% off in APAC")
            ],
            conflicts=[],
            sources=[SimpleNamespace(source_id="W1", page=page, kind="press")],
            citations=[WEB_SOURCE],
            warnings=[],
            queries=queries or ["competitor promotion APAC"],
            providers_used=["fixtures"],
            reason=None,
        )


@pytest.fixture
def agent_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        env="test",
        llm_provider="fake",
        agent_retry_base_delay_s=0,
        sandbox_backend="subprocess",
        sandbox_timeout_s=20,
        artifacts_dir=str(tmp_path / "artifacts"),
    )


def make_toolbox(
    llm: FakeLLM,
    settings: Settings,
    *,
    sql: FakeSQL | None = None,
    documents: FakeDocs | None = None,
    web: FakeWeb | None = None,
    analysis: bool = False,
    breaker: CircuitBreaker | None = None,
) -> Toolbox:
    return Toolbox(
        llm=llm,
        settings=settings,
        sql=cast(SQLTool, sql),
        documents=cast(DocumentTool, documents),
        web=cast(WebResearchTool, web),
        analysis=AnalysisTool(llm, settings) if analysis else None,
        breaker=breaker or CircuitBreaker(),
    )


def service(toolbox: Toolbox) -> AgentService:
    return AgentService(build_graph(toolbox, memory_checkpointer()))


async def ask(toolbox: Toolbox, question: str, user: UserContext = FINANCE) -> RunResult:
    return await service(toolbox).ask(question, user)
