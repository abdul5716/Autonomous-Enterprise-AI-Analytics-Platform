from datetime import date

import pytest

from analytics_platform.config import Settings
from analytics_platform.security import UserContext
from analytics_platform.tools.rag.chunking import chunk_document
from analytics_platform.tools.rag.embeddings import HashingEmbedder
from analytics_platform.tools.rag.models import Chunk, DocumentMeta, RetrievedChunk
from analytics_platform.tools.rag.parse import ParsedSection
from analytics_platform.tools.rag.rerank import OverlapReranker
from analytics_platform.tools.rag.retrieval import HybridRetriever


class InMemoryStore:
    """Mirrors PgDocumentStore's filters (ACL and status) without a database."""

    def __init__(self) -> None:
        self.embedder = HashingEmbedder()
        self.items: list[tuple[DocumentMeta, Chunk, list[float]]] = []

    def add(self, meta: DocumentMeta, sections: dict[str, list[str]]) -> None:
        parsed = [ParsedSection(heading, 1, blocks) for heading, blocks in sections.items()]
        for chunk in chunk_document(meta, parsed):
            vector = self.embedder.embed_documents([chunk.embedding_text(meta)])[0]
            self.items.append((meta, chunk, vector))

    def _visible(self, user: UserContext, include_superseded: bool) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                **chunk.model_dump(exclude={"chunk_index"}),
                title=meta.title,
                doc_type=meta.doc_type,
                status=meta.status,
                effective_date=meta.effective_date,
                supersedes=meta.supersedes,
            )
            for meta, chunk, _ in self.items
            if user.can_read(meta.acl) and (include_superseded or meta.status == "current")
        ]

    def vector_search(
        self, embedding: list[float], user: UserContext, k: int, include_superseded: bool
    ) -> list[RetrievedChunk]:
        vectors = {c.chunk_id: v for _, c, v in self.items}
        scored = [
            c.model_copy(
                update={
                    "score": sum(a * b for a, b in zip(embedding, vectors[c.chunk_id], strict=True))
                }
            )
            for c in self._visible(user, include_superseded)
        ]
        return sorted(scored, key=lambda c: c.score, reverse=True)[:k]

    def text_search(
        self, query: str, user: UserContext, k: int, include_superseded: bool
    ) -> list[RetrievedChunk]:
        words = {w for w in query.lower().split() if len(w) > 3}
        scored = []
        for c in self._visible(user, include_superseded):
            hits = sum(w in c.content.lower() for w in words)
            if hits:
                scored.append(c.model_copy(update={"score": float(hits)}))
        return sorted(scored, key=lambda c: c.score, reverse=True)[:k]


def meta(doc_id: str, version: str = "1.0", **kwargs: object) -> DocumentMeta:
    values: dict[str, object] = {
        "doc_id": doc_id,
        "title": f"{doc_id} title",
        "doc_type": "policy",
        "version": version,
        "effective_date": date(2026, 1, 1),
        "owner": "Test",
        "acl": ["all"],
        "status": "current",
        "filename": f"{doc_id}.pdf",
    }
    values.update(kwargs)
    return DocumentMeta.model_validate(values)


@pytest.fixture
def store() -> InMemoryStore:
    s = InMemoryStore()
    s.add(
        meta("POL-FIN-007", "3.0", title="Refund and Cancellation Policy", supersedes="v2.0"),
        {
            "4. Enterprise customers": [
                "Enterprise customers may request a pro-rated refund within 30 days of the "
                "invoice date.",
                "Any single refund above $50,000 requires approval from the VP of Finance.",
            ]
        },
    )
    s.add(
        meta("POL-FIN-007", "2.0", title="Refund and Cancellation Policy", status="superseded"),
        {"2. Refunds": ["Refunds are available only for billing errors, within 14 days."]},
    )
    s.add(
        meta("QBR-Q2", title="Quarterly Business Review", acl=["leadership", "finance"]),
        {"Outlook and risks": ["Key risks: the EMEA price increase may raise churn."]},
    )
    s.add(
        meta("HR-POL-004", title="Travel and Expense Policy"),
        {"Reimbursement": ["Economy class is required for flights under 6 hours."]},
    )
    return s


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def retriever(store: InMemoryStore, settings: Settings) -> HybridRetriever:
    return HybridRetriever(store, HashingEmbedder(), OverlapReranker(), settings)
