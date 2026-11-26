from datetime import date
from typing import Literal

from pydantic import BaseModel

from analytics_platform.tools.base import Citation


class DocumentMeta(BaseModel):
    """Document-management metadata (from the manifest), stored alongside every chunk."""

    doc_id: str
    title: str
    doc_type: str
    version: str
    effective_date: date
    owner: str
    acl: list[str]
    status: Literal["current", "superseded", "draft"]
    supersedes: str | None = None
    filename: str

    @property
    def ref(self) -> str:
        return f"{self.doc_id} v{self.version}"


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    version: str
    chunk_index: int
    section: str
    page: int
    content: str

    def embedding_text(self, meta: DocumentMeta) -> str:
        """Title and section prefix, so a chunk like 'Up to $5,000 | ...' keeps its context."""
        return f"{meta.title} ({meta.doc_type}) - {self.section}\n{self.content}"


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    version: str
    title: str
    doc_type: str
    status: str
    effective_date: date
    supersedes: str | None
    section: str
    page: int
    content: str
    score: float = 0.0  # retriever-specific until reranked, then a 0-1 relevance
    retrievers: list[str] = []

    @property
    def ref(self) -> str:
        return f"{self.doc_id} v{self.version}"

    def citation(self) -> Citation:
        detail = f"{self.section}, page {self.page}"
        if self.status != "current":
            detail += f" ({self.status})"
        return Citation(source_type="document", source_id=self.ref, title=self.title, detail=detail)
