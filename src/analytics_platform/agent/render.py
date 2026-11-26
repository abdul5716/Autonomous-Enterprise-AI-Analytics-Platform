"""Turn the synthesiser's draft into the final answer: numbered sources, tables, charts, report."""

import re
from pathlib import Path

from analytics_platform.agent.models import (
    Answer,
    ContractItem,
    DraftAnswer,
    Evidence,
    ReportLinks,
    TableData,
)
from analytics_platform.security import UserContext
from analytics_platform.tools.base import Citation
from analytics_platform.tools.report.models import (
    Report,
    ReportChart,
    ReportSection,
    ReportTable,
    check_report,
)
from analytics_platform.tools.report.render import save_report

# A run of adjacent references such as [S1][S4] or [S1, S2], numbered together without repeats.
EVIDENCE_REFS = re.compile(r"(?:\[S\w+(?:\s*,\s*S\w+)*\])+")
STEP_ID = re.compile(r"S\w+")
SPACE_BEFORE_PUNCTUATION = re.compile(r"[ \t]+(?=[.,;:!?)])")
REPEATED_SPACES = re.compile(r"(?<=\S)[ \t]{2,}")


class Numbering:
    """Replaces [S1] evidence references with [1][2] numbered sources, in order of first use."""

    def __init__(self, evidence: dict[str, Evidence]) -> None:
        self.evidence = evidence
        self.citations: list[Citation] = []
        self._index: dict[tuple[str, str], int] = {}
        self.cited_steps: list[str] = []
        self.unknown: set[str] = set()

    def apply(self, text: str) -> str:
        text = EVIDENCE_REFS.sub(self._replace, text)
        text = SPACE_BEFORE_PUNCTUATION.sub("", text)
        return REPEATED_SPACES.sub(" ", text).strip()

    def _replace(self, match: re.Match[str]) -> str:
        numbers: list[int] = []
        for step_id in STEP_ID.findall(match.group(0)):
            ev = self.evidence.get(step_id)
            if ev is None or not ev.found:
                self.unknown.add(step_id)
                continue
            if step_id not in self.cited_steps:
                self.cited_steps.append(step_id)
            for c in ev.citations:
                key = (c.source_type, c.source_id)
                if key not in self._index:
                    self.citations.append(c)
                    self._index[key] = len(self.citations)
                if self._index[key] not in numbers:
                    numbers.append(self._index[key])
        return "".join(f"[{n}]" for n in numbers)


def contract_gaps(
    contract: list[ContractItem], draft: DraftAnswer, evidence: dict[str, Evidence]
) -> list[str]:
    """Contract items the draft does not address with evidence that was actually found."""
    coverage = {c.item_id: c for c in draft.coverage}
    gaps = []
    for item in contract:
        cov = coverage.get(item.id)
        supported = (
            cov is not None
            and cov.addressed
            and any(evidence.get(sid) is not None and evidence[sid].found for sid in cov.evidence)
        )
        if not supported:
            gaps.append(f"Not answered: {item.description}")
    return gaps


def build_answer(
    draft: DraftAnswer,
    evidence: dict[str, Evidence],
    *,
    gaps: list[str],
    assumptions: list[str],
) -> Answer:
    numbering = Numbering(evidence)
    text = numbering.apply(draft.answer)
    findings = [numbering.apply(f) for f in draft.key_findings]
    warnings = [w for ev in evidence.values() for w in ev.warnings]
    if numbering.unknown:
        warnings.append(
            f"Removed references to missing evidence: {', '.join(sorted(numbering.unknown))}."
        )
    found = [ev for ev in evidence.values() if ev.found]
    return Answer(
        status="partial" if gaps else "complete",
        text=text,
        key_findings=findings,
        tables=[ev.table for ev in found if ev.table is not None and ev.tool != "chart"],
        charts=[ev.chart for ev in found if ev.chart is not None],
        citations=numbering.citations,
        assumptions=assumptions,
        gaps=gaps,
        warnings=list(dict.fromkeys(warnings)),
        confidence="medium" if gaps and draft.confidence == "high" else draft.confidence,
    )


def _report_table(table: TableData) -> ReportTable:
    return ReportTable(columns=table.columns, rows=table.rows, caption=table.title)


def build_report(
    draft: DraftAnswer,
    answer: Answer,
    evidence: dict[str, Evidence],
    *,
    question: str,
    user: UserContext,
) -> Report:
    numbering = Numbering(evidence)  # the report numbers only the sources its own text cites
    summary = [numbering.apply(f) for f in draft.key_findings] or [numbering.apply(draft.answer)]
    sections = []
    for s in draft.sections:
        paragraphs = [numbering.apply(p) for p in s.paragraphs]
        shown = [evidence[sid] for sid in s.evidence if sid in evidence and evidence[sid].found]
        chart = next((ev.chart for ev in shown if ev.chart), None)
        table = next((ev.table for ev in shown if ev.table), None)
        sections.append(
            ReportSection(
                heading=s.heading,
                paragraphs=paragraphs,
                table=_report_table(table) if table and not chart else None,
                chart=ReportChart(spec=chart.spec, data=_report_table(chart.data))
                if chart
                else None,
            )
        )
    if not sections:
        sections = [ReportSection(heading="Answer", paragraphs=[numbering.apply(draft.answer)])]
    return Report(
        title=draft.title or "Analysis report",
        question=question,
        prepared_for=user.display_name,
        executive_summary=summary,
        sections=sections,
        citations=numbering.citations,
        assumptions=answer.assumptions,
        limitations=answer.gaps,
    )


def publish_report(report: Report, artifacts_dir: str) -> tuple[ReportLinks, list[str]]:
    problems = check_report(report)
    files = save_report(report, Path(artifacts_dir))
    links = ReportLinks(report_id=files.report_id, html=str(files.html), pdf=str(files.pdf))
    return links, [f"Report check: {p}" for p in problems]
