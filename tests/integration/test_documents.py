"""Needs `make up` and `make data` (which also indexes documents)."""

import pytest

from analytics_platform.config import get_settings
from analytics_platform.llm import FakeLLM
from analytics_platform.security import DEMO_USERS
from analytics_platform.tools.rag.tool import DocumentTool, build_document_tool

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tool() -> DocumentTool:
    return build_document_tool(FakeLLM(), get_settings())


@pytest.mark.parametrize(
    ("user", "visible", "hidden"),
    [
        ("employee", "POL-FIN-007 v3.0", "MEMO-PRC-2026-04 v1.0"),
        ("emea.sales", "MEMO-PRC-2026-04 v1.0", "QBR-Q2-2026 v1.0"),
    ],
)
async def test_row_level_security_on_documents(
    tool: DocumentTool, user: str, visible: str, hidden: str
) -> None:
    refs = set()
    for query in ["refund policy", "EMEA Professional price increase", "quarterly business review"]:
        r = await tool.search(query, DEMO_USERS[user])
        refs |= {c.ref for c in r.chunks}

    assert visible in refs
    assert not any(ref.startswith(hidden.split(" ")[0]) for ref in refs)


async def test_enterprise_refund_question_finds_the_current_policy(tool: DocumentTool) -> None:
    r = await tool.search(
        "What is the refund policy for enterprise customers?", DEMO_USERS["employee"]
    )

    assert r.evidence == "strong"
    assert (r.chunks[0].ref, r.chunks[0].section) == ("POL-FIN-007 v3.0", "4. Enterprise customers")
    assert all(c.status == "current" for c in r.chunks)


async def test_unrelated_question_has_no_evidence(tool: DocumentTool) -> None:
    r = await tool.search("What is the parental leave policy?", DEMO_USERS["finance.admin"])

    assert r.evidence == "none"
