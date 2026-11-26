"""pgvector + Postgres full-text storage for chunks, with document ACLs enforced by RLS."""

import re
from collections.abc import Sequence
from typing import Any, Protocol

import psycopg
from psycopg.rows import dict_row

from analytics_platform.config import Settings
from analytics_platform.security import UserContext
from analytics_platform.tools.base import ErrorKind, ToolError
from analytics_platform.tools.rag.models import Chunk, DocumentMeta, RetrievedChunk

WORD = re.compile(r"[a-z0-9]+")


class ChunkStore(Protocol):
    def vector_search(
        self, embedding: list[float], user: UserContext, k: int, include_superseded: bool
    ) -> list[RetrievedChunk]: ...

    def text_search(
        self, query: str, user: UserContext, k: int, include_superseded: bool
    ) -> list[RetrievedChunk]: ...


def schema_ddl(dim: int, rag_password: str) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS docs.documents (
    doc_id          text NOT NULL,
    version         text NOT NULL,
    title           text NOT NULL,
    doc_type        text NOT NULL,
    owner           text NOT NULL,
    status          text NOT NULL CHECK (status IN ('current', 'superseded', 'draft')),
    effective_date  date NOT NULL,
    supersedes      text,
    acl             text[] NOT NULL,
    filename        text NOT NULL,
    sha256          text NOT NULL,
    ingested_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, version)
);
COMMENT ON COLUMN docs.documents.acl IS 'Groups allowed to read; ''all'' means every employee';

CREATE TABLE IF NOT EXISTS docs.chunks (
    chunk_id     text PRIMARY KEY,
    doc_id       text NOT NULL,
    version      text NOT NULL,
    chunk_index  int NOT NULL,
    section      text NOT NULL,
    page         int NOT NULL,
    content      text NOT NULL,
    embedding    vector({dim}) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (
                     setweight(to_tsvector('english', section), 'A')
                     || setweight(to_tsvector('english', content), 'B')
                 ) STORED,
    FOREIGN KEY (doc_id, version) REFERENCES docs.documents ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON docs.chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON docs.chunks USING gin (tsv);

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag_reader') THEN
        CREATE ROLE rag_reader LOGIN PASSWORD '{rag_password}';
    END IF;
END
$$;
ALTER ROLE rag_reader SET default_transaction_read_only = on;
ALTER ROLE rag_reader SET statement_timeout = '10s';
GRANT CONNECT ON DATABASE nimbus TO rag_reader;
GRANT USAGE ON SCHEMA docs TO rag_reader;
GRANT SELECT ON docs.documents, docs.chunks TO rag_reader;

-- app.user_groups holds the caller's groups. Unset or empty: only documents for 'all'.
CREATE OR REPLACE FUNCTION docs.acl_visible(acl text[]) RETURNS boolean
LANGUAGE sql STABLE AS $fn$
    SELECT 'all' = ANY(acl) OR coalesce(
        acl && string_to_array(nullif(current_setting('app.user_groups', true), ''), ','),
        false
    )
$fn$;

ALTER TABLE docs.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE docs.chunks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS acl ON docs.documents;
DROP POLICY IF EXISTS acl ON docs.chunks;
CREATE POLICY acl ON docs.documents FOR SELECT TO rag_reader USING (docs.acl_visible(acl));
CREATE POLICY acl ON docs.chunks FOR SELECT TO rag_reader USING (EXISTS (
    SELECT 1 FROM docs.documents d WHERE d.doc_id = chunks.doc_id AND d.version = chunks.version
));
"""


def _vector(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"


def or_tsquery(text: str) -> str | None:
    """Match any query word (ranked by how many match), unlike websearch_to_tsquery's AND."""
    words = sorted(set(WORD.findall(text.lower())))
    return " | ".join(words) or None


_SELECT = """
    SELECT c.chunk_id, c.doc_id, c.version, d.title, d.doc_type, d.status, d.effective_date,
           d.supersedes, c.section, c.page, c.content, {score} AS score
    FROM docs.chunks c
    JOIN docs.documents d ON d.doc_id = c.doc_id AND d.version = c.version
    WHERE (%(superseded)s OR d.status = 'current') {where}
    ORDER BY score DESC
    LIMIT %(k)s
"""


class PgDocumentStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # --- ingestion (owner role) ---

    def ensure_schema(self) -> None:
        ddl = schema_ddl(
            self.settings.embedding_dim, self.settings.postgres_rag_password.get_secret_value()
        )
        with psycopg.connect(self.settings.postgres_dsn()) as conn:
            conn.execute(ddl)

    def existing_hashes(self) -> dict[tuple[str, str], str]:
        with psycopg.connect(self.settings.postgres_dsn()) as conn:
            rows = conn.execute("SELECT doc_id, version, sha256 FROM docs.documents").fetchall()
        return {(r[0], r[1]): r[2] for r in rows}

    def upsert(
        self,
        meta: DocumentMeta,
        sha256: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        with psycopg.connect(self.settings.postgres_dsn()) as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM docs.documents WHERE doc_id = %s AND version = %s",
                (meta.doc_id, meta.version),
            )
            cur.execute(
                "INSERT INTO docs.documents (doc_id, version, title, doc_type, owner, status, "
                "effective_date, supersedes, acl, filename, sha256) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    meta.doc_id,
                    meta.version,
                    meta.title,
                    meta.doc_type,
                    meta.owner,
                    meta.status,
                    meta.effective_date,
                    meta.supersedes,
                    meta.acl,
                    meta.filename,
                    sha256,
                ),
            )
            cur.executemany(
                "INSERT INTO docs.chunks (chunk_id, doc_id, version, chunk_index, section, page, "
                "content, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)",
                [
                    (
                        c.chunk_id,
                        c.doc_id,
                        c.version,
                        c.chunk_index,
                        c.section,
                        c.page,
                        c.content,
                        _vector(e),
                    )
                    for c, e in zip(chunks, embeddings, strict=True)
                ],
            )

    def delete_missing(self, keep: set[tuple[str, str]]) -> int:
        with psycopg.connect(self.settings.postgres_dsn()) as conn:
            rows = conn.execute("SELECT doc_id, version FROM docs.documents").fetchall()
            stale = [(r[0], r[1]) for r in rows if (r[0], r[1]) not in keep]
            for doc_id, version in stale:
                conn.execute(
                    "DELETE FROM docs.documents WHERE doc_id = %s AND version = %s",
                    (doc_id, version),
                )
        return len(stale)

    # --- retrieval (rag_reader role, RLS applies) ---

    def _search(
        self, score: str, where: str, params: dict[str, Any], user: UserContext
    ) -> list[RetrievedChunk]:
        sql = _SELECT.format(score=score, where=where)
        try:
            with (
                psycopg.connect(self.settings.postgres_rag_dsn(), connect_timeout=3) as conn,
                conn.cursor(row_factory=dict_row) as cur,
            ):
                cur.execute(
                    "SELECT set_config('app.user_groups', %s, true)", (",".join(user.groups),)
                )
                cur.execute(sql, params)
                return [RetrievedChunk(**row) for row in cur.fetchall()]
        except psycopg.errors.QueryCanceled as exc:
            raise ToolError("Document search timed out", kind=ErrorKind.TIMEOUT) from exc
        except psycopg.OperationalError as exc:
            raise ToolError(
                f"Document store is unavailable: {exc}", kind=ErrorKind.UNAVAILABLE
            ) from exc

    def vector_search(
        self, embedding: list[float], user: UserContext, k: int, include_superseded: bool
    ) -> list[RetrievedChunk]:
        return self._search(
            "1 - (c.embedding <=> %(q)s::vector)",
            "",
            {"q": _vector(embedding), "k": k, "superseded": include_superseded},
            user,
        )

    def text_search(
        self, query: str, user: UserContext, k: int, include_superseded: bool
    ) -> list[RetrievedChunk]:
        tsquery = or_tsquery(query)
        if tsquery is None:
            return []
        return self._search(
            "ts_rank_cd(c.tsv, to_tsquery('english', %(q)s))",
            "AND c.tsv @@ to_tsquery('english', %(q)s)",
            {"q": tsquery, "k": k, "superseded": include_superseded},
            user,
        )
