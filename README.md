# Autonomous Enterprise AI Analytics Platform

An agentic analytics and research platform. Business users ask questions in natural language and get
reliable, evidence-backed answers: text, tables, charts or full reports, drawn from SQL databases,
internal documents and the public web.

It handles requests like:

- "Show me total revenue by region for the last six months."
- "Revenue dropped 15% last quarter. Find the contributing regions, analyze why, and chart it."
- "Compare our product performance with three competitors and prepare a report."
- "What is our refund policy for enterprise customers?"

The company in the demo, **Nimbus Analytics**, is fictional. Its data is synthetic, with patterns
planted on purpose so the agent has real findings to discover. Its competitors are real companies, so
market research uses genuine public information.

See [docs/architecture.md](docs/architecture.md) for the component and workflow diagrams.

## Status

| Milestone | Status |
|---|---|
| 1. Foundation: stack, config, LLM client, CI | Done |
| 2. Synthetic data: Postgres CRM, Oracle finance, DuckDB warehouse, PDFs | Done |
| 3. Tools: semantic layer + text-to-SQL, RAG, web research, sandbox, charts | In progress (SQL and documents done) |
| 4. LangGraph agent: planning, execution, retries, fallbacks, checkpoints | Planned |
| 5. Validation and completion judge | Planned |
| 6. FastAPI + Streamlit UI | Planned |
| 7. Evaluations and observability | Planned |
| 8. Demos and polish | Planned |

## Tech stack

| Concern | Choice |
|---|---|
| Orchestration | LangGraph |
| LLM access | LiteLLM (any provider), default `gpt-4o-mini` |
| Structured data | PostgreSQL 16, Oracle Database Free 23ai, DuckDB |
| Document search | pgvector + Postgres full-text, fused with RRF; local `fastembed` embeddings and `flashrank` reranking |
| Cache | Redis |
| Observability | Arize Phoenix (OpenTelemetry) |
| UI / API | Streamlit, FastAPI |
| Tooling | uv, ruff, mypy (strict), pytest, GitHub Actions |

## Quickstart

Requires Docker and [uv](https://docs.astral.sh/uv/).

```bash
make setup     # install dependencies, create .env from .env.example
# edit .env and set OPENAI_API_KEY (or GEMINI_API_KEY plus the LLM_*_MODEL variables)
make up        # start Postgres, Oracle, Redis, Phoenix
make health    # check every service; add an LLM ping with:
uv run python -m analytics_platform.healthcheck --llm
make data      # generate and load all synthetic data, then index the PDFs (~15 seconds;
               # the first run also downloads ~100 MB of embedding/reranking models)
make check     # lint, type check, unit tests (offline)
uv run pytest -m integration   # tests against the running databases
```

## Synthetic data

`make data` simulates about three years of Nimbus Analytics history, relative to today, and loads it
into three systems, the way an enterprise's data is actually spread out:

| System | Contents |
|---|---|
| Postgres `crm` schema | customers, subscriptions, price history, sales opportunities, support tickets |
| Oracle `finance` schema | invoices, refunds and SLA credits, monthly budgets |
| DuckDB warehouse | star schema built by an ETL from the two systems above: `fact_revenue`, `fact_budget`, and dimensions for date, region, product and customer |
| `data/documents/` | 15 PDFs: policies, memos, an incident postmortem, a quarterly review, product guides |

The last complete quarter contains a planted revenue drop of about 15%. It is driven by EMEA (a
Professional edition price increase) and APAC (a platform outage and a competitor promotion). The
evidence is spread across all of the systems above, so finding the causes takes SQL, documents, or
both. All document figures are computed from the data, so documents and databases always agree.

The expected answers are written to `data/generated/answer_key.json` for evaluations. The agent's
tools never read that file.

Host ports: Postgres `55432`, Oracle `51521`, Redis `56379`, Phoenix UI
[localhost:6006](http://localhost:6006).

## Tools

Each tool enforces the caller's permissions itself (the demo users are in
`src/analytics_platform/security.py`), returns citations, and raises classified errors
(`transient`, `invalid_input`, `permission`, `timeout`, `unavailable`, ...) so the agent can decide
whether to retry, repair, fall back or report.

**SQL** (`tools/sql/`). Questions are first mapped to governed metrics in
`semantic_layer/metrics.yaml` (net revenue, refunds, budget, ...), which compile to parameterised
warehouse SQL. Relative periods ("last 6 months", "last quarter") resolve against the warehouse's
data date, and every interpretation is reported back as an assumption. Questions the metrics can't
express fall back to LLM-written SQL on any of the three databases, which must pass a `sqlglot`
guard: one `SELECT`, allowlisted tables, no file/session/sleep functions, enforced row limit. The
databases are a second line of defence: read-only roles, statement timeouts, and Postgres
row-level security that limits a regional user to their region's rows.

**Documents** (`tools/rag/`). PDFs are parsed into sections using font sizes, chunked by section,
embedded locally and stored in pgvector with their ACL, version and status. Retrieval runs vector
and keyword search, fuses them with reciprocal rank fusion, and reranks with a cross-encoder.
Postgres row-level security filters chunks by the user's groups, and superseded versions are
excluded unless asked for. A three-level evidence check (strong / weak / none) skips the LLM
entirely when nothing relevant is accessible. Answers must cite passage ids with word-for-word
quotes, which are verified; unverifiable claims are repaired once, then dropped.

## Cost

Built to run for a few dollars in total:

- Embeddings and reranking run locally on CPU.
- LLM responses at temperature 0 are cached in Redis, so repeated demos are nearly free.
- Model tiers send routing and classification to the cheapest model.
- CI and unit tests use `FakeLLM` and never call a paid API.

## Project layout

```
src/analytics_platform/
  config.py        settings from environment / .env
  healthcheck.py   service connectivity check
  llm/             LLM client: tiers, retries, fallback, cache, usage tracking, FakeLLM
  security.py      user context: groups, region scope, demo users
  tools/sql/       semantic layer compiler, period resolver, SQL guard, data sources, SQL tool
  tools/rag/       PDF parsing, chunking, embeddings, pgvector store, hybrid retrieval, Q&A
semantic_layer/    governed metric and dimension definitions
src/data_gen/      synthetic data: simulation, Postgres/Oracle loaders, warehouse ETL, PDFs
infra/postgres/    database init (schemas, pgvector, read-only analyst role)
docs/              architecture and design decisions
tests/             unit tests (offline); tests/integration needs the compose stack
```
