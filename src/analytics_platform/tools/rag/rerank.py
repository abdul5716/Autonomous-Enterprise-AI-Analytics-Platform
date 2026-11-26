"""Cross-encoder reranking: scores each (question, chunk) pair, giving a 0-1 relevance."""

from typing import Protocol

from analytics_platform.config import Settings
from analytics_platform.tools.rag.models import RetrievedChunk


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Return the chunks sorted by relevance, with `score` set to a 0-1 relevance."""
        ...


class FlashRankReranker:
    """ms-marco-MiniLM-L-12-v2 on CPU (~22 MB)."""

    def __init__(self, settings: Settings) -> None:
        from flashrank import Ranker

        self._ranker = Ranker(
            model_name=settings.reranker_model, cache_dir=settings.model_cache_dir
        )

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        from flashrank import RerankRequest

        if not chunks:
            return []
        passages = [
            {"id": i, "text": f"{c.title} - {c.section}\n{c.content}"} for i, c in enumerate(chunks)
        ]
        results = self._ranker.rerank(RerankRequest(query=query, passages=passages))
        return [
            chunks[int(r["id"])].model_copy(update={"score": float(r["score"])}) for r in results
        ]


class OverlapReranker:
    """Fraction of query words found in the chunk. Deterministic; for tests and fallback."""

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        words = {w for w in query.lower().split() if len(w) > 3}
        scored = []
        for c in chunks:
            text = f"{c.title} {c.section} {c.content}".lower()
            score = sum(w in text for w in words) / len(words) if words else 0.0
            scored.append(c.model_copy(update={"score": score}))
        return sorted(scored, key=lambda c: c.score, reverse=True)
