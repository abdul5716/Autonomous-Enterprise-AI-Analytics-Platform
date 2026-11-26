from datetime import date
from pathlib import Path

from analytics_platform.tools.rag.chunking import chunk_document
from analytics_platform.tools.rag.parse import ParsedSection, parse_pdf
from data_gen.documents import Document, Section, render_pdf
from tests.rag.conftest import meta


def test_parse_pdf_recovers_sections_paragraphs_and_tables(tmp_path: Path) -> None:
    doc = Document(
        doc_id="POL-T-1",
        title="Test Policy",
        doc_type="policy",
        version="1.0",
        effective_date=date(2026, 1, 1),
        owner="Test",
        acl=["all"],
        status="current",
        sections=[
            Section("1. Scope", ["First paragraph.", "Second <b>bold</b> paragraph."]),
            Section("2. Matrix", ["Approvals:"], table=[["Amount", "Approver"], ["$5", "CSM"]]),
        ],
    )
    path = tmp_path / doc.filename
    render_pdf(doc, path)

    parsed = parse_pdf(path)

    assert parsed.title == "Test Policy"
    assert [s.heading for s in parsed.sections] == ["1. Scope", "2. Matrix"]
    assert parsed.sections[0].blocks == ["First paragraph.", "Second bold paragraph."]
    assert parsed.sections[1].blocks == ["Approvals:", "| Amount | Approver |\n| $5 | CSM |"]


def test_short_sections_become_one_chunk_each() -> None:
    sections = [ParsedSection("A", 1, ["one", "two"]), ParsedSection("B", 2, ["three"])]

    chunks = chunk_document(meta("D-1"), sections)

    assert [(c.chunk_id, c.section, c.page, c.content) for c in chunks] == [
        ("D-1@1.0#0", "A", 1, "one\ntwo"),
        ("D-1@1.0#1", "B", 2, "three"),
    ]


def test_long_sections_split_on_paragraphs_with_overlap() -> None:
    paragraphs = [f"Paragraph {i} " + "x" * 50 for i in range(5)]

    chunks = chunk_document(meta("D-1"), [ParsedSection("Long", 1, paragraphs)], max_chars=150)

    assert len(chunks) > 1
    assert all(len(c.content) <= 150 for c in chunks)
    first_end = chunks[0].content.split("\n")[-1]
    assert chunks[1].content.startswith(first_end)  # last paragraph repeated as overlap


def test_tables_are_never_split_or_repeated() -> None:
    table = "| a | b |\n| 1 | 2 |"
    paragraphs = ["x" * 100, table, "y" * 100]

    chunks = chunk_document(meta("D-1"), [ParsedSection("T", 1, paragraphs)], max_chars=130)

    assert sum(table in c.content for c in chunks) == 1


def test_embedding_text_carries_title_and_section() -> None:
    m = meta("D-1", title="Refund Policy")
    chunk = chunk_document(m, [ParsedSection("5. Approval matrix", 1, ["| Up to $5,000 |"])])[0]

    assert chunk.embedding_text(m).startswith("Refund Policy (policy) - 5. Approval matrix")
