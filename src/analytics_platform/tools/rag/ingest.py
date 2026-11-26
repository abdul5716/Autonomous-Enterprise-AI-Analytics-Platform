"""Index the document folder: `python -m analytics_platform.tools.rag.ingest [--force]`.

Reads `manifest.json` (the document-management metadata: ACL, version, status), parses each
PDF into sections, chunks, embeds and upserts. Unchanged files (same SHA-256) are skipped.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from analytics_platform.config import Settings, get_settings
from analytics_platform.tools.rag.chunking import chunk_document
from analytics_platform.tools.rag.embeddings import Embedder, FastEmbedEmbedder
from analytics_platform.tools.rag.models import DocumentMeta
from analytics_platform.tools.rag.parse import parse_pdf
from analytics_platform.tools.rag.store import PgDocumentStore


@dataclass
class IngestReport:
    indexed: list[str]
    skipped: list[str]
    removed: int
    chunks: int


def load_manifest(documents_dir: Path) -> list[DocumentMeta]:
    raw = json.loads((documents_dir / "manifest.json").read_text())
    return [DocumentMeta.model_validate(entry) for entry in raw]


def ingest(
    settings: Settings, store: PgDocumentStore, embedder: Embedder, *, force: bool = False
) -> IngestReport:
    documents_dir = Path(settings.documents_dir)
    store.ensure_schema()
    existing = store.existing_hashes()
    report = IngestReport(indexed=[], skipped=[], removed=0, chunks=0)
    keep: set[tuple[str, str]] = set()
    for meta in load_manifest(documents_dir):
        path = documents_dir / meta.filename
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        key = (meta.doc_id, meta.version)
        keep.add(key)
        if not force and existing.get(key) == sha256:
            report.skipped.append(meta.ref)
            continue
        parsed = parse_pdf(path)
        if not parsed.sections:
            raise ValueError(f"No sections found in {path}; is it a scanned PDF?")
        chunks = chunk_document(meta, parsed.sections)
        embeddings = embedder.embed_documents([c.embedding_text(meta) for c in chunks])
        store.upsert(meta, sha256, chunks, embeddings)
        report.indexed.append(meta.ref)
        report.chunks += len(chunks)
    report.removed = store.delete_missing(keep)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-index unchanged documents")
    args = parser.parse_args()
    settings = get_settings()
    report = ingest(
        settings, PgDocumentStore(settings), FastEmbedEmbedder(settings), force=args.force
    )
    print(
        f"Indexed {len(report.indexed)} documents ({report.chunks} chunks), "
        f"skipped {len(report.skipped)} unchanged, removed {report.removed} stale."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
