"""Turn a PDF into sections using font sizes: headings are large, tables are small."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pypdf import PdfReader

TITLE_SIZE = 17.0
HEADING_SIZE = 13.0
TABLE_SIZE = 9.5  # table cells (and the metadata block) are set smaller than body text
PARAGRAPH_GAP = 15.0  # vertical gap (pt) that separates body paragraphs; line leading is 12


@dataclass
class ParsedSection:
    heading: str
    page: int
    blocks: list[str] = field(default_factory=list)  # paragraphs, or a whole table

    @property
    def text(self) -> str:
        return "\n".join(self.blocks)


@dataclass
class ParsedDocument:
    title: str
    pages: int
    sections: list[ParsedSection]


@dataclass
class _Line:
    page: int
    y: float
    size: float
    cells: list[str]


def _lines(reader: PdfReader) -> list[_Line]:
    lines: list[_Line] = []
    for page_no, page in enumerate(reader.pages, start=1):
        current: dict[float, _Line] = {}

        def visit(
            text: str,
            cm: list[float],
            tm: list[float],
            font: dict[str, Any] | None,
            size: float,
            page_no: int = page_no,
            current: dict[float, _Line] = current,
        ) -> None:
            if not text.strip():
                return
            y = round(tm[5] * cm[3] + cm[5], 1)
            line = current.get(y)
            if line is None:
                line = current[y] = _Line(page_no, y, size * tm[0], [])
                lines.append(line)
            line.size = max(line.size, size * tm[0])
            line.cells.append(text.replace("\n", ""))

        page.extract_text(visitor_text=visit)
    return lines


def parse_pdf(path: Path) -> ParsedDocument:
    reader = PdfReader(path)
    title = ""
    sections: list[ParsedSection] = []
    previous_y: float | None = None
    for line in _lines(reader):
        if line.size >= TITLE_SIZE:
            title = " ".join(c.strip() for c in line.cells)
            continue
        if line.size >= HEADING_SIZE:
            sections.append(ParsedSection(" ".join(c.strip() for c in line.cells), line.page))
            previous_y = None
            continue
        if not sections:
            continue  # letterhead and metadata block; metadata comes from the manifest
        section = sections[-1]
        if line.size <= TABLE_SIZE:
            row = " | ".join(c.strip() for c in line.cells)
            if section.blocks and section.blocks[-1].startswith("| "):
                section.blocks[-1] += f"\n| {row} |"
            else:
                section.blocks.append(f"| {row} |")
            previous_y = None
            continue
        text = "".join(line.cells).strip()
        new_paragraph = (
            not section.blocks
            or section.blocks[-1].startswith("| ")
            or (previous_y is not None and previous_y - line.y > PARAGRAPH_GAP)
        )
        if new_paragraph:
            section.blocks.append(text)
        else:
            section.blocks[-1] += " " + text
        previous_y = line.y
    return ParsedDocument(title=title, pages=len(reader.pages), sections=sections)
