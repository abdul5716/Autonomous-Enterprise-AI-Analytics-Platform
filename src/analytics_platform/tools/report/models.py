import re
import uuid
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from analytics_platform.tools.base import Citation
from analytics_platform.tools.charts import ChartSpec, validate_chart

MARKER = re.compile(r"\[(\d+)\]")
MAX_TABLE_ROWS = 40


class ReportTable(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    caption: str | None = None

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, caption: str | None = None) -> "ReportTable":
        rows = frame.astype(object).where(frame.notna(), None).values.tolist()
        return cls(columns=[str(c) for c in frame.columns], rows=rows, caption=caption)


class ReportChart(BaseModel):
    spec: ChartSpec
    data: ReportTable

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.data.rows, columns=self.data.columns)


class ReportSection(BaseModel):
    heading: str
    paragraphs: list[str] = []
    bullets: list[str] = []
    table: ReportTable | None = None
    chart: ReportChart | None = None


class Report(BaseModel):
    report_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    title: str
    question: str
    prepared_for: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    executive_summary: list[str]
    sections: list[ReportSection]
    citations: list[Citation]  # [n] in text refers to citations[n - 1]
    assumptions: list[str] = []
    limitations: list[str] = []

    def texts(self) -> list[str]:
        out = list(self.executive_summary)
        for s in self.sections:
            out += s.paragraphs + s.bullets
        return out


def check_report(report: Report) -> list[str]:
    """Structural problems: dangling citation markers, unused sources, invalid charts."""
    problems = []
    used: set[int] = set()
    for text in report.texts():
        for n in map(int, MARKER.findall(text)):
            if not 1 <= n <= len(report.citations):
                problems.append(f"Citation [{n}] does not exist ({len(report.citations)} sources).")
            used.add(n)
    unused = [i for i in range(1, len(report.citations) + 1) if i not in used]
    if unused:
        problems.append(f"Sources {unused} are listed but never cited.")
    if not report.executive_summary:
        problems.append("The report has no executive summary.")
    for section in report.sections:
        if not (section.paragraphs or section.bullets or section.table or section.chart):
            problems.append(f"Section '{section.heading}' is empty.")
        if section.chart:
            problems += [
                f"Chart in '{section.heading}': {p}"
                for p in validate_chart(section.chart.spec, section.chart.frame())
            ]
    return problems
