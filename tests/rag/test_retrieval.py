from datetime import date

import pytest

from analytics_platform.config import Settings
from analytics_platform.security import DEMO_USERS
from analytics_platform.tools.rag.embeddings import HashingEmbedder
from analytics_platform.tools.rag.models import RetrievedChunk
from analytics_platform.tools.rag.rerank import OverlapReranker
from analytics_platform.tools.rag.retrieval import HybridRetriever, reciprocal_rank_fusion
from analytics_platform.tools.rag.store import or_tsquery
from tests.rag.conftest import InMemoryStore

ADMIN = DEMO_USERS["finance.admin"]
EMPLOYEE = DEMO_USERS["employee"]


def chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="D",
        version="1",
        title="t",
        doc_type="policy",
        status="current",
        effective_date=date(2026, 1, 1),
        supersedes=None,
        section="s",
        page=1,
        content="c",
    )


def test_rrf_rewards_chunks_found_by_both_retrievers() -> None:
    fused = reciprocal_rank_fusion(
        {"vector": [chunk("a"), chunk("b")], "keyword": [chunk("b"), chunk("c")]}
    )

    assert [c.chunk_id for c in fused] == ["b", "a", "c"]
    assert fused[0].retrievers == ["vector", "keyword"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_or_tsquery_keeps_only_safe_words() -> None:
    assert or_tsquery("Refund policy: who approves $50,000?") == (
        "000 | 50 | approves | policy | refund | who"
    )
    assert or_tsquery("?!") is None


async def test_finds_current_policy(retriever: HybridRetriever) -> None:
    r = await retriever.retrieve("enterprise refund approval", EMPLOYEE)

    assert r.evidence == "strong"
    assert r.chunks[0].ref == "POL-FIN-007 v3.0"
    assert all(c.status == "current" for c in r.chunks)


async def test_superseded_versions_only_on_request(retriever: HybridRetriever) -> None:
    r = await retriever.retrieve("refunds billing errors", EMPLOYEE, include_superseded=True)

    assert r.chunks[0].ref == "POL-FIN-007 v2.0"
    assert r.chunks[0].citation().detail == "2. Refunds, page 1 (superseded)"


async def test_acl_hides_restricted_documents(retriever: HybridRetriever) -> None:
    hidden = await retriever.retrieve("risks churn price increase", EMPLOYEE)
    visible = await retriever.retrieve("risks churn price increase", ADMIN)

    assert "QBR-Q2 v1.0" not in {c.ref for c in hidden.chunks}
    assert visible.chunks[0].ref == "QBR-Q2 v1.0"


async def test_no_evidence_for_unrelated_question(retriever: HybridRetriever) -> None:
    r = await retriever.retrieve("parental leave entitlement", ADMIN)

    assert r.evidence == "none"
    assert not r.sufficient


class FixedReranker:
    def __init__(self, score: float) -> None:
        self.score = score

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return [c.model_copy(update={"score": self.score}) for c in chunks]


@pytest.mark.parametrize(("score", "evidence"), [(0.9, "strong"), (0.1, "weak"), (0.001, "none")])
async def test_evidence_bands(
    store: InMemoryStore, settings: Settings, score: float, evidence: str
) -> None:
    retriever = HybridRetriever(store, HashingEmbedder(), FixedReranker(score), settings)

    r = await retriever.retrieve("enterprise refund", ADMIN)

    assert r.evidence == evidence


class BrokenEmbedder(HashingEmbedder):
    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("model file missing")


class BrokenReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        raise RuntimeError("onnx crashed")


async def test_degrades_to_keyword_search_when_embeddings_fail(
    store: InMemoryStore, settings: Settings
) -> None:
    retriever = HybridRetriever(store, BrokenEmbedder(), OverlapReranker(), settings)

    r = await retriever.retrieve("enterprise refund approval", EMPLOYEE)

    assert r.chunks[0].ref == "POL-FIN-007 v3.0"
    assert r.chunks[0].retrievers == ["keyword"]
    assert "keyword search only" in r.warnings[0]


async def test_falls_back_when_reranker_fails(store: InMemoryStore, settings: Settings) -> None:
    retriever = HybridRetriever(store, HashingEmbedder(), BrokenReranker(), settings)

    r = await retriever.retrieve("enterprise refund approval", EMPLOYEE)

    assert r.chunks[0].ref == "POL-FIN-007 v3.0"
    assert "Reranking was unavailable" in r.warnings[0]
