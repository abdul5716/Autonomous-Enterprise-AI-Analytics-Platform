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
| 3. Tools: semantic layer + text-to-SQL, RAG, web research, sandbox, charts | Planned |
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
| Vector search | pgvector + BM25, local `fastembed` embeddings |
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
make data      # generate and load all synthetic data (~10 seconds)
make check     # lint, type check, tests
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
src/data_gen/      synthetic data: simulation, Postgres/Oracle loaders, warehouse ETL, PDFs
infra/postgres/    database init (schemas, pgvector, read-only analyst role)
docs/              architecture and design decisions
tests/             unit tests (offline)
```
