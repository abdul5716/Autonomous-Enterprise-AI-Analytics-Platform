"""Agent state: what was asked, what "done" means, the plan, and the evidence gathered."""

import operator
import time
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

from analytics_platform.config import Settings
from analytics_platform.security import UserContext
from analytics_platform.tools.base import Citation
from analytics_platform.tools.charts import ChartSpec

RequestType = Literal["data_query", "root_cause", "market_research", "document_qa", "out_of_scope"]
OutputFormat = Literal["text", "table", "chart", "report"]
ToolName = Literal["sql", "documents", "web", "analysis", "chart"]
StepStatus = Literal["pending", "done", "failed", "skipped"]
TABLE_TOOLS: frozenset[str] = frozenset({"sql", "analysis"})


class Intent(BaseModel):
    request_type: RequestType
    output_format: OutputFormat = Field(
        description="report only if a report/document is requested; chart if a chart/plot is"
    )
    goal: str = Field(description="The request restated as one precise sentence")
    ambiguity: str | None = Field(
        None, description="What is unclear such that no sensible default exists; else null"
    )
    clarifying_question: str | None = None


class ContractItem(BaseModel):
    """One thing the final answer must contain; together they define "done"."""

    id: str = Field(description="C1, C2, ...")
    description: str


class PlanStep(BaseModel):
    id: str = Field(description="S1, S2, ...")
    tool: ToolName
    input: str = Field(description="Question or instruction for the tool, specific to this step")
    depends_on: list[str] = []
    covers: list[str] = Field(default=[], description="Contract item ids this step provides")
    queries: list[str] = Field(default=[], description="web only: up to 3 public search queries")


class Plan(BaseModel):
    contract: list[ContractItem] = Field(min_length=1)
    steps: list[PlanStep] = Field(min_length=1)


class StepRun(BaseModel):
    step_id: str
    status: StepStatus = "pending"
    attempts: int = 0
    error: str | None = None
    error_kind: str | None = None
    fallback: str | None = None
    duration_s: float = 0.0


class TableData(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    title: str
    truncated: bool = False


class ChartData(BaseModel):
    spec: ChartSpec
    vega_lite: dict[str, Any]
    data: TableData


class Evidence(BaseModel):
    """A tool result reduced to what the synthesiser reads, plus what the UI shows."""

    step_id: str
    tool: ToolName
    found: bool
    summary: str
    table: TableData | None = None
    chart: ChartData | None = None
    citations: list[Citation] = []
    assumptions: list[str] = []
    warnings: list[str] = []
    detail: dict[str, Any] = {}


class Budget(BaseModel):
    max_tool_calls: int
    max_llm_calls: int
    max_cost_usd: float
    max_seconds: float
    max_replans: int
    started_at: float = Field(default_factory=time.time)
    tool_calls: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0
    replans: int = 0

    @classmethod
    def from_settings(cls, settings: Settings) -> "Budget":
        return cls(
            max_tool_calls=settings.agent_max_tool_calls,
            max_llm_calls=settings.agent_max_llm_calls,
            max_cost_usd=settings.agent_max_cost_usd,
            max_seconds=settings.agent_max_seconds,
            max_replans=settings.agent_max_replans,
        )

    def exhausted(self) -> str | None:
        if self.tool_calls >= self.max_tool_calls:
            return f"tool call budget ({self.max_tool_calls}) used up"
        if self.llm_calls >= self.max_llm_calls:
            return f"LLM call budget ({self.max_llm_calls}) used up"
        if self.cost_usd >= self.max_cost_usd:
            return f"cost budget (${self.max_cost_usd:.2f}) used up"
        if time.time() - self.started_at >= self.max_seconds:
            return f"time budget ({self.max_seconds:.0f}s) used up"
        return None


class Coverage(BaseModel):
    item_id: str
    addressed: bool
    evidence: list[str] = Field(default=[], description="Step ids whose evidence addresses it")


class DraftSection(BaseModel):
    heading: str
    paragraphs: list[str] = Field(min_length=1)
    evidence: list[str] = Field(default=[], description="Step ids whose table or chart to show")


class DraftAnswer(BaseModel):
    answer: str = Field(description="Markdown answer; cite evidence like [S1] after each claim")
    key_findings: list[str] = Field(default=[], description="3-5 one-sentence findings, cited")
    coverage: list[Coverage]
    confidence: Literal["high", "medium", "low"]
    title: str | None = Field(None, description="Report title (report format only)")
    sections: list[DraftSection] = Field(default=[], description="Report sections (report only)")


class ReportLinks(BaseModel):
    report_id: str
    html: str
    pdf: str


class Answer(BaseModel):
    status: Literal["complete", "partial", "failed", "refused"]
    text: str
    key_findings: list[str] = []
    tables: list[TableData] = []
    charts: list[ChartData] = []
    citations: list[Citation] = []
    assumptions: list[str] = []
    gaps: list[str] = []
    warnings: list[str] = []
    confidence: Literal["high", "medium", "low"] = "medium"
    report: ReportLinks | None = None


class TraceEvent(BaseModel):
    node: str
    message: str
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentState(TypedDict, total=False):
    question: str
    user: UserContext
    clarifications: list[str]
    intent: Intent
    contract: list[ContractItem]
    plan: list[PlanStep]
    plan_problems: list[str]
    runs: dict[str, StepRun]
    evidence: dict[str, Evidence]
    assumptions: list[str]
    gaps: list[str]
    budget: Budget
    draft: DraftAnswer | None
    answer: Answer
    trace: Annotated[list[TraceEvent], operator.add]
