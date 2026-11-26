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
| 3. Tools: semantic layer + text-to-SQL, RAG, web research, sandbox, charts, reports | Done |
| 4. LangGraph agent: planning, execution, retries, fallbacks, checkpoints | Done |
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
make sandbox   # build the isolated Python analysis image
make ask Q="Why did revenue drop last quarter?"   # ask the agent (AS=emea.sales to switch user)
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
| `data/web_fixtures/` | offline web snapshot on `.example` domains: competitor press release, outage news, analyst report, illustrative vendor profiles, a forum thread, and a blog with planted prompt injection |

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

**Web research** (`tools/web/`). The LLM drafts search queries, and an egress guard refuses any
query containing internal document ids, record ids, money amounts or confidentiality markers.
Search tries Tavily (if `TAVILY_API_KEY` is set), then free engines via `ddgs`, then an offline
snapshot written by `make data` (`WEB_MODE=fixtures` uses only the snapshot, for evals and CI).
Pages are fetched with SSRF protection on every redirect (no private, loopback or cloud-metadata
addresses) and a size cap, and the main text is extracted with `trafilatura`. All web content is
untrusted: hidden elements are stripped, and sentences addressed to an AI ("ignore previous
instructions", "you are now ...") are removed and flagged, which halves the source's credibility.
Each source is scored on credibility (vendor, analyst, press, reference, blog, forum), recency
and relevance. Findings must quote their sources word for word, like document answers. Searches
and pages are cached in Redis for a day, and a page that can't be fetched falls back to its
search snippet, marked as such.

**Python analysis** (`tools/sandbox/`). For calculations SQL can't express (growth decomposition,
correlations, statistics), the LLM writes pandas code that runs in a throwaway Docker container
with no network, a read-only filesystem, a non-root user, no capabilities, and memory, CPU,
process and time limits. Before anything runs, an AST check allows only analysis libraries
(pandas, numpy, scipy, standard maths) and rejects file, process and introspection access. Data
goes in on stdin and the result comes back on stdout. Failed code is repaired up to twice using
the error message.

**Charts** (`tools/charts.py`). Chart specs are validated against the data's columns, and a chart
type is suggested from its shape: lines for time series, grouped bars for current-vs-previous
comparisons. Specs compile to Vega-Lite, which is shown interactively in the UI and rendered to
PNG/SVG for reports.

**Reports** (`tools/report/`). A report is summary, sections, tables, charts and numbered sources.
A checker rejects citations to missing sources, unused sources and empty sections. Reports render
to HTML (autoescaped, charts inline as SVG) and PDF, and are saved under `data/artifacts/reports/`.

## Agent

The agent (`agent/`) is a LangGraph graph. It works in five stages:

1. It classifies the request as a data query, root-cause analysis, market research, document
   question or out of scope.
2. If the request can't be answered sensibly as stated, it pauses with one clarifying question,
   then resumes from its Postgres checkpoint once the user replies. Missing time periods or
   regions don't count: defaults are applied and reported as assumptions.
3. Simple requests get a fixed one-tool plan with no planner call. Everything else is planned by
   the LLM as a dependency graph of tool steps, together with an **answer contract**: the items a
   complete answer must contain.
4. A deterministic checker validates the plan: known tools, earlier dependencies only, charts
   and analyses fed by tables, no internal data in web queries, and every contract item covered.
   A bad plan is sent back once with the problems; invalid steps that remain are dropped and
   recorded as gaps.
5. The executor runs independent steps in parallel. A synthesiser then writes the answer from the
   evidence only, citing step ids, which become numbered sources. Reports are rendered to
   HTML/PDF.

Failures degrade the answer instead of breaking it:

- Transient errors are retried with backoff.
- A database that keeps failing trips a circuit breaker.
- If Oracle is down, SQL falls back to the other sources.
- Steps whose inputs failed are skipped.
- Per-request budgets (tool calls, LLM calls, cost, time) stop runaway work.

Whatever could not be done is listed as a gap in a partial answer. See
[docs/architecture.md](docs/architecture.md#the-graph-as-built) for the graph.

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
  agent/           LangGraph graph, planning checks, step execution, rendering, checkpoints, CLI
  reliability/     retries with backoff, circuit breaker
  tools/sql/       semantic layer compiler, period resolver, SQL guard, data sources, SQL tool
  tools/rag/       PDF parsing, chunking, embeddings, pgvector store, hybrid retrieval, Q&A
  tools/web/       search providers, safe fetching, sanitiser, source scoring, research tool
  tools/sandbox/   code checks, isolated runner, Docker/subprocess backends, analysis tool
  tools/charts.py  chart specs, suggestions, Vega-Lite rendering
  tools/report/    report model, citation checks, HTML/PDF rendering
semantic_layer/    governed metric and dimension definitions
src/data_gen/      synthetic data: simulation, Postgres/Oracle loaders, warehouse ETL, PDFs
infra/postgres/    database init (schemas, pgvector, read-only analyst role)
infra/sandbox/     analysis sandbox image
docs/              architecture and design decisions
tests/             unit tests (offline); tests/integration needs the compose stack
```
