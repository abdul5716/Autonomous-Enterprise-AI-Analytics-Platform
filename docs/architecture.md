# Architecture

## Platform components

```mermaid
flowchart LR
    U[Business user] --> UI[Streamlit UI]
    UI --> GW[FastAPI gateway<br/>auth, RBAC, rate limit]
    GW --> ORCH

    subgraph ORCH[Agent orchestrator - LangGraph]
        IA[Intent and ambiguity analyzer]
        PL[Planner]
        EX[Executor]
        VA[Validators]
        SY[Synthesizer and report builder]
        CJ[Completion judge]
    end

    ORCH <--> ST[(Postgres<br/>checkpoints, audit log)]
    ORCH --> LLMGW[LiteLLM<br/>model routing, retries, fallback, cache]

    subgraph TOOLS[Tools]
        SQL[Text-to-SQL]
        RAG[Document retrieval]
        WEB[Web research]
        PY[Python sandbox]
        CH[Charts and reports]
    end

    EX --> TOOLS
    SQL --> SEM[Semantic layer<br/>metrics, fiscal calendar]
    SEM --> PG[(Postgres CRM)]
    SEM --> ORA[(Oracle finance)]
    SEM --> DW[(DuckDB warehouse)]
    RAG --> VEC[(pgvector + BM25<br/>with ACL metadata)]
    WEB --> EXT[Search APIs]
    ORCH --> OBS[Phoenix traces]
```

## Agent workflow

```mermaid
flowchart TD
    START([User question]) --> CTX[Load session context<br/>role, history, permissions]
    CTX --> INT[Intent analysis<br/>goal, metrics, time range, output type]
    INT --> AMB{Ambiguous?}

    AMB -- "High-impact" --> CLAR[Ask clarifying question]
    CLAR -->|User replies, resume from checkpoint| INT
    AMB -- "Low-impact" --> ASSUME[Apply default, record assumption]
    AMB -- Clear --> ROUTE
    ASSUME --> ROUTE

    ROUTE{Route}
    ROUTE -- "Single KB lookup" --> FAST_RAG[Fast path: RAG]
    ROUTE -- "Single metric" --> FAST_SQL[Fast path: SQL]
    ROUTE -- "Multi-step" --> PLAN

    FAST_RAG --> STEPVAL
    FAST_SQL --> STEPVAL

    PLAN[Planner: step DAG<br/>and answer contract] --> PCHK{Plan check<br/>permissions, budget}
    PCHK -- Rejected --> PLAN
    PCHK -- Approved --> NEXT

    NEXT[Executor: next ready steps,<br/>parallel where independent] --> TOOL[Invoke tool]
    TOOL --> RES{Outcome}

    RES -- "Transient error" --> RETRY[Backoff retry, max 3]
    RETRY --> TOOL
    RES -- "Semantic error" --> REPAIR[Self-correct, max 2]
    REPAIR --> TOOL
    RES -- "Persistent failure" --> FB{Fallback?}
    FB -- Yes --> ALT[Alternate tool or source]
    ALT --> TOOL
    FB -- No --> GAP[Record data gap]
    RES -- Success --> STEPVAL

    STEPVAL{Step validation} -- Fail --> REPAIR
    STEPVAL -- Pass --> SAVE[Save evidence]
    GAP --> SAVE

    SAVE --> MORE{More steps?}
    MORE -- Yes --> NEXT
    MORE -- No --> SYN

    SYN[Synthesize answer] --> OUTVAL{Output validation<br/>numbers, citations,<br/>groundedness, PII}
    OUTVAL -- "Fail, revisions left" --> SYN
    OUTVAL -- Pass --> DONE

    DONE{Completion judge:<br/>contract satisfied?}
    DONE -- "Gaps, budget left" --> REPLAN[Replan missing items]
    REPLAN --> PCHK
    DONE -- "Gaps, budget exhausted" --> PARTIAL[Partial answer with gaps]
    DONE -- Complete --> RENDER

    PARTIAL --> RENDER
    RENDER[Render: text, table, chart, report] --> DELIVER([Deliver with sources,<br/>assumptions, SQL, confidence])
```

## Key decisions

- **The completion judge is deterministic.** An "answer contract" (a checklist of what a complete answer
  needs) is created during intent analysis. The task is done when every item is backed by validated
  evidence, or recorded as a gap once the budget is spent. The LLM saying "done" is not enough.
- **A semantic layer sits between the LLM and the databases.** Metrics like revenue are defined once,
  so the SQL agent composes governed definitions instead of guessing join logic.
- **LLM calls go through one client** (`analytics_platform.llm`): model tiers (fast, strong, judge),
  retries on transient errors only, fallback models, a Redis response cache, and per-run cost tracking
  via `track_usage()`. Tests use `FakeLLM`, so CI never calls a paid API.
- **The agent's database role is read-only** (`analyst_ro`): enforced by Postgres, not by prompts.
