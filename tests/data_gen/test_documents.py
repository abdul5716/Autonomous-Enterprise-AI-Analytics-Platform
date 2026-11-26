import json
from datetime import date
from pathlib import Path

import pytest
from pypdf import PdfReader

from data_gen.documents import Document, write_documents
from data_gen.simulate import Dataset, simulate


@pytest.fixture(scope="module")
def ds() -> Dataset:
    return simulate(date(2026, 11, 26), seed=42, n_customers=300)


@pytest.fixture(scope="module")
def written(ds: Dataset, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[Document]]:
    out = tmp_path_factory.mktemp("docs")
    return out, write_documents(ds, out)


def pdf_text(path: Path) -> str:
    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


def test_every_document_has_a_pdf_and_manifest_entry(written: tuple[Path, list[Document]]) -> None:
    out, docs = written
    manifest = json.loads((out / "manifest.json").read_text())

    assert len(manifest) == len(docs) == 15
    assert all((out / entry["filename"]).exists() for entry in manifest)
    assert all(entry["acl"] for entry in manifest)


def test_refund_policy_versions(written: tuple[Path, list[Document]]) -> None:
    out, docs = written
    refund = {d.version: d for d in docs if d.doc_id == "POL-FIN-007"}

    assert refund["3.0"].status == "current"
    assert refund["2.0"].status == "superseded"
    text = pdf_text(out / refund["3.0"].filename)
    assert "pro-rated refund" in text
    assert "30 days" in text
    assert "$50,000" in text


def test_pricing_memo_matches_price_book(ds: Dataset, written: tuple[Path, list[Document]]) -> None:
    out, docs = written
    memo = next(d for d in docs if d.doc_id.startswith("MEMO-PRC"))
    new_price = ds.prices.price("P02", "EMEA", ds.timeline.drop_q_start)

    assert f"${new_price:,.2f}" in pdf_text(out / memo.filename)
    assert "finance" in memo.acl and "all" not in memo.acl


def test_postmortem_cites_actual_sla_credit_total(
    ds: Dataset, written: tuple[Path, list[Document]]
) -> None:
    out, docs = written
    postmortem = next(d for d in docs if d.doc_type == "postmortem")
    total = ds.refunds[ds.refunds.reason == "sla_credit"].amount_usd.sum()

    assert f"${total:,.0f}" in pdf_text(out / postmortem.filename)
