import json
from collections.abc import Sequence

from analytics_platform.llm import FakeLLM, Message
from analytics_platform.security import DEMO_USERS
from analytics_platform.tools.rag.retrieval import HybridRetriever
from analytics_platform.tools.rag.tool import Claim, DocumentTool, check_claims

EMPLOYEE = DEMO_USERS["employee"]
V3 = "POL-FIN-007@3.0#0"
V2 = "POL-FIN-007@2.0#0"


def answer(*claims: tuple[str, str], answerable: bool = True) -> str:
    return json.dumps(
        {
            "answerable": answerable,
            "answer": "Enterprise customers get pro-rated refunds within 30 days.",
            "claims": [
                {"statement": "s", "chunk_ids": [cid], "quote": quote} for cid, quote in claims
            ],
            "missing": None if answerable else "Nothing about this.",
        }
    )


GOOD = (V3, "may request a pro-rated refund within 30 days")


async def test_answer_with_verified_citations(retriever: HybridRetriever) -> None:
    llm = FakeLLM().when("Passages:", answer(GOOD))

    result = await DocumentTool(llm, retriever).ask("enterprise refund policy", EMPLOYEE)

    assert result.answered
    assert [c.source_id for c in result.citations] == ["POL-FIN-007 v3.0"]
    assert result.citations[0].detail == "4. Enterprise customers, page 1"
    assert result.warnings == []


async def test_passages_are_labelled_with_ids_and_versions(retriever: HybridRetriever) -> None:
    llm = FakeLLM().when("Passages:", answer(GOOD))

    await DocumentTool(llm, retriever).ask("enterprise refund policy", EMPLOYEE)

    system = llm.calls[0].messages[1]["content"]
    assert f"[{V3}] Refund and Cancellation Policy (POL-FIN-007 v3.0" in system


async def test_misquoted_claim_is_repaired(retriever: HybridRetriever) -> None:
    llm = (
        FakeLLM()
        .when("Fix these citation problems", answer(GOOD))
        .when("Passages:", answer((V3, "refunds are always available for 90 days")))
    )

    result = await DocumentTool(llm, retriever).ask("enterprise refund policy", EMPLOYEE)

    assert result.answered and len(llm.calls) == 2
    assert "does not appear word for word" in llm.calls[1].messages[-1]["content"]


async def test_unverifiable_claims_are_dropped(retriever: HybridRetriever) -> None:
    llm = FakeLLM().when("Passages:", answer(GOOD, ("POL-X@1#0", "made up")))

    result = await DocumentTool(llm, retriever).ask("enterprise refund policy", EMPLOYEE)

    assert result.answered
    assert len(result.claims) == 1
    assert "Dropped 1 claim(s)" in result.warnings[0]


async def test_no_llm_call_without_evidence(retriever: HybridRetriever) -> None:
    llm = FakeLLM()  # any call would raise

    result = await DocumentTool(llm, retriever).ask("parental leave entitlement", EMPLOYEE)

    assert not result.answered
    assert result.reason is not None and "None of the documents" in result.reason
    assert llm.calls == []


async def test_model_can_decline(retriever: HybridRetriever) -> None:
    llm = FakeLLM().when("Passages:", answer(answerable=False))

    result = await DocumentTool(llm, retriever).ask("enterprise refund currency", EMPLOYEE)

    assert not result.answered
    assert result.reason == "Nothing about this."


async def test_citing_a_superseded_version_is_flagged(retriever: HybridRetriever) -> None:
    def respond(messages: Sequence[Message]) -> str:
        assert "[SUPERSEDED]" in messages[1]["content"]
        return answer((V2, "only for billing errors, within 14 days"))

    llm = FakeLLM().when("Passages:", respond)

    result = await DocumentTool(llm, retriever).ask(
        "old refunds policy billing errors", EMPLOYEE, include_superseded=True
    )

    assert result.citations[0].detail == "2. Refunds, page 1 (superseded)"
    assert "superseded document version" in result.warnings[-1]


def test_quote_matching_ignores_case_whitespace_and_markup(retriever: HybridRetriever) -> None:
    chunks = {c.chunk_id: c for c in retriever.store._visible(EMPLOYEE, False)}  # type: ignore[attr-defined]
    claim = Claim(statement="s", chunk_ids=[V3], quote="ANY single refund   above **$50,000**.")

    assert check_claims([claim], chunks) == {}
