"""Hybrid retrieval: vector + keyword search, fused with RRF, reranked, then a sufficiency check."""

import asyncio
import logging
from typing import Literal

from pydantic import BaseModel

from analytics_platform.config import Settings
from analytics_platform.security import UserContext
from analytics_platform.tools.rag.embeddings import Embedder
from analytics_platform.tools.rag.models import RetrievedChunk
from analytics_platform.tools.rag.rerank import OverlapReranker, Reranker
from analytics_platform.tools.rag.store import ChunkStore

logger = logging.getLogger(__name__)

RRF_K = 60  # standard constant from the RRF paper; damps the advantage of rank 1


class Retrieval(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    evidence: Literal["strong", "weak", "none"]
    reason: str | None = None
    warnings: list[str] = []

    @property
    def sufficient(self) -> bool:
        return self.evidence != "none"


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[RetrievedChunk]], k: int = RRF_K
) -> list[RetrievedChunk]:
    """Combine rankings without comparing their incompatible raw scores."""
    scores: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    found_by: dict[str, list[str]] = {}
    for name, chunks in ranked_lists.items():
        for rank, chunk in enumerate(chunks, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            best.setdefault(chunk.chunk_id, chunk)
            found_by.setdefault(chunk.chunk_id, []).append(name)
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [
        best[cid].model_copy(update={"score": scores[cid], "retrievers": found_by[cid]})
        for cid in ordered
    ]


class HybridRetriever:
    def __init__(
        self, store: ChunkStore, embedder: Embedder, reranker: Reranker, settings: Settings
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = settings.rag_candidates
        self.top_k = settings.rag_top_k
        self.min_relevance = settings.rag_min_relevance
        self.strong_relevance = settings.rag_strong_relevance

    async def retrieve(
        self,
        query: str,
        user: UserContext,
        *,
        k: int | None = None,
        include_superseded: bool = False,
    ) -> Retrieval:
        warnings: list[str] = []
        ranked: dict[str, list[RetrievedChunk]] = {}
        try:
            embedding = await asyncio.to_thread(self.embedder.embed_query, query)
        except Exception as exc:  # model files missing or corrupt: keep keyword search alive
            logger.warning("Embedding failed, using keyword search only: %s", exc)
            warnings.append("Semantic search was unavailable; results use keyword search only.")
        else:
            ranked["vector"] = await asyncio.to_thread(
                self.store.vector_search, embedding, user, self.candidates, include_superseded
            )
        ranked["keyword"] = await asyncio.to_thread(
            self.store.text_search, query, user, self.candidates, include_superseded
        )
        fused = reciprocal_rank_fusion(ranked)[: self.candidates]
        try:
            reranked = await asyncio.to_thread(self.reranker.rerank, query, fused)
        except Exception as exc:
            logger.warning("Reranker failed, using word-overlap ranking: %s", exc)
            warnings.append("Reranking was unavailable; relevance scores are approximate.")
            reranked = OverlapReranker().rerank(query, fused)
        chunks = [c for c in reranked if c.score >= self.min_relevance / 2][: k or self.top_k]
        evidence, reason = self._sufficiency(chunks)
        return Retrieval(
            query=query, chunks=chunks, evidence=evidence, reason=reason, warnings=warnings
        )

    def _sufficiency(
        self, chunks: list[RetrievedChunk]
    ) -> tuple[Literal["strong", "weak", "none"], str | None]:
        top = chunks[0] if chunks else None
        if top is None or top.score < self.min_relevance:
            return "none", "None of the documents you can access covers this question."
        if top.score < self.strong_relevance:
            return "weak", (
                f"The closest passage ({top.ref}, '{top.section}') is only loosely related "
                f"(relevance {top.score:.2f})."
            )
        return "strong", None
