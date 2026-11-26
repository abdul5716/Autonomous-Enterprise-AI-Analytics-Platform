"""Structure-aware chunking: one chunk per section, split only when a section is long."""

from analytics_platform.tools.rag.models import Chunk, DocumentMeta
from analytics_platform.tools.rag.parse import ParsedSection

MAX_CHARS = 1200


def _pack(blocks: list[str], max_chars: int) -> list[str]:
    """Group whole paragraphs/tables into pieces; repeat the last paragraph as overlap."""
    pieces: list[list[str]] = [[]]
    for block in blocks:
        size = sum(len(b) + 1 for b in pieces[-1])
        if pieces[-1] and size + len(block) > max_chars:
            overlap = pieces[-1][-1:] if not pieces[-1][-1].startswith("| ") else []
            if overlap and len(overlap[0]) + len(block) > max_chars:
                overlap = []
            pieces.append([*overlap])
        pieces[-1].append(block)
    return ["\n".join(p) for p in pieces if p]


def chunk_document(
    meta: DocumentMeta, sections: list[ParsedSection], max_chars: int = MAX_CHARS
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        if not section.blocks:
            continue
        for content in _pack(section.blocks, max_chars):
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{meta.doc_id}@{meta.version}#{index}",
                    doc_id=meta.doc_id,
                    version=meta.version,
                    chunk_index=index,
                    section=section.heading,
                    page=section.page,
                    content=content,
                )
            )
    return chunks
