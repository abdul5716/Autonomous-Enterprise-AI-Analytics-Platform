from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from analytics_platform.tools.base import Citation
from analytics_platform.tools.charts import ChartSpec
from analytics_platform.tools.report.models import (
    Report,
    ReportChart,
    ReportSection,
    ReportTable,
    check_report,
)
from analytics_platform.tools.report.render import render_html, save_report

FRAME = pd.DataFrame(
    {
        "region": ["NA", "EMEA"],
        "net_revenue": [21.9e6, 10.7e6],
        "net_revenue_change_pct": [-3.9, -28.7],
    }
)
CITATIONS = [
    Citation(
        source_type="sql", source_id="sql:abc", title="Warehouse query", detail="SELECT " * 60
    ),
    Citation(
        source_type="web",
        source_id="W1",
        title="<script>alert(1)</script> news",
        url="https://news.example/a",
    ),
]


def report(**overrides: object) -> Report:
    values: dict[str, object] = {
        "title": "Revenue review",
        "question": "Why did revenue fall?",
        "prepared_for": "Fiona",
        "executive_summary": ["EMEA fell 28.7% [1].", "A competitor is discounting [2]."],
        "sections": [
            ReportSection(
                heading="By region",
                paragraphs=["See the chart [1]."],
                table=ReportTable.from_frame(FRAME, caption="USD"),
                chart=ReportChart(
                    spec=ChartSpec(type="bar", title="Net revenue", x="region", y=["net_revenue"]),
                    data=ReportTable.from_frame(FRAME),
                ),
            )
        ],
        "citations": CITATIONS,
        "assumptions": ["Last quarter means 2026-Q3."],
    }
    values.update(overrides)
    return Report.model_validate(values)


def test_valid_report_has_no_problems() -> None:
    assert check_report(report()) == []


def test_dangling_and_unused_citations_are_reported() -> None:
    problems = check_report(report(executive_summary=["Claim [3]."]))

    assert "Citation [3] does not exist (2 sources)." in problems
    assert "Sources [2] are listed but never cited." in problems


def test_empty_sections_and_bad_charts_are_reported() -> None:
    bad_chart = ReportChart(
        spec=ChartSpec(type="line", title="t", x="region", y=["net_revenue"]),
        data=ReportTable.from_frame(FRAME),
    )
    problems = check_report(
        report(
            sections=[ReportSection(heading="Empty"), ReportSection(heading="C", chart=bad_chart)]
        )
    )

    assert "Section 'Empty' is empty." in problems
    assert any("needs a time x axis" in p for p in problems)


def test_html_escapes_content_and_links_citations() -> None:
    html = render_html(report())

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert '<a class="cite" href="#source-1">[1]</a>' in html
    assert "<svg" in html
    assert "-28.7%" in html and "21,900,000" in html
    assert "SELECT SELECT" in html and "SELECT " * 40 not in html  # long details shortened


def test_unsafe_urls_are_not_linked() -> None:
    bad = Citation(source_type="web", source_id="W9", title="x", url="javascript:alert(1)")

    html = render_html(report(citations=[CITATIONS[0], bad]))

    assert 'href="javascript' not in html


def test_save_writes_html_pdf_and_json(tmp_path: Path) -> None:
    files = save_report(report(), tmp_path)

    text = "\n".join(page.extract_text() for page in PdfReader(files.pdf).pages)
    assert "Revenue review" in text and "EMEA fell 28.7% [1]." in text
    assert "Sources" in text and "https://news.example/a" in text
    assert files.html.read_text().startswith("<!doctype html>")
    assert Report.model_validate_json(files.json_path.read_text()).report_id == files.report_id
