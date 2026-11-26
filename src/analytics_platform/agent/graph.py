"""The agent graph: understand, (clarify), plan, check, execute in parallel waves, answer."""

import asyncio
from datetime import date
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from analytics_platform.agent.models import (
    AgentState,
    Answer,
    Budget,
    DraftAnswer,
    Evidence,
    Intent,
    Plan,
    StepRun,
    TraceEvent,
)
from analytics_platform.agent.planning import check_plan, ensure_chart, fast_plan, prune
from analytics_platform.agent.prompts import (
    INTENT_SYSTEM,
    PLANNER_SYSTEM,
    REPORT_RULES,
    SYNTHESIS_SYSTEM,
)
from analytics_platform.agent.render import (
    build_answer,
    build_report,
    contract_gaps,
    publish_report,
)
from analytics_platform.agent.steps import execute_step
from analytics_platform.agent.toolbox import Toolbox
from analytics_platform.llm import Message, ModelTier, Usage, track_usage
from analytics_platform.security import UserContext

Node = Literal[
    "intent", "clarify", "refuse", "fast_plan", "planner", "plan_check", "executor",
    "synthesize", "render",
]  # fmt: skip

REFUSAL = (
    "I can help with questions about Nimbus Analytics' business: company data, internal "
    "documents, and public market information. This request is outside that scope."
)


def _event(node: str, message: str) -> list[TraceEvent]:
    return [TraceEvent(node=node, message=message)]


def _charge(budget: Budget, usage: Usage) -> Budget:
    return budget.model_copy(
        update={
            "llm_calls": budget.llm_calls + usage.calls,
            "cost_usd": budget.cost_usd + usage.cost_usd,
        }
    )


def _user_line(user: UserContext) -> str:
    scope = ", ".join(user.region_scope) if user.region_scope else "all regions"
    return f"{user.display_name} (groups: {', '.join(user.groups) or 'none'}; sees {scope})"


def request_text(state: AgentState) -> str:
    """The question plus any clarification answers, as the tools should see it."""
    extra = state.get("clarifications") or []
    return state["question"] + ("\n" + "\n".join(extra) if extra else "")


def build_graph(
    toolbox: Toolbox, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    llm, settings = toolbox.llm, toolbox.settings

    async def intent_node(state: AgentState) -> dict[str, Any]:
        messages: list[Message] = [
            {
                "role": "system",
                "content": INTENT_SYSTEM.format(
                    today=date.today().isoformat(), user=_user_line(state["user"])
                ),
            },
            {"role": "user", "content": request_text(state)},
        ]
        with track_usage() as usage:
            intent = await llm.structured(messages, Intent, tier=ModelTier.FAST)
        budget = state.get("budget") or Budget.from_settings(settings)
        return {
            "intent": intent,
            "budget": _charge(budget, usage),
            "trace": _event("intent", f"{intent.request_type} -> {intent.output_format}: "
                            f"{intent.goal}"),
        }  # fmt: skip

    def route_intent(state: AgentState) -> Node:
        intent = state["intent"]
        if intent.request_type == "out_of_scope":
            return "refuse"
        asked = len(state.get("clarifications") or [])
        if (
            intent.ambiguity
            and intent.clarifying_question
            and asked < (settings.agent_max_clarifications)
        ):
            return "clarify"
        if fast_plan(intent, state["question"]) is not None:
            return "fast_plan"
        return "planner"

    def clarify_node(state: AgentState) -> dict[str, Any]:
        intent = state["intent"]
        reply = interrupt({"question": intent.clarifying_question, "ambiguity": intent.ambiguity})
        return {
            "clarifications": [
                *(state.get("clarifications") or []),
                f"Clarification - Q: {intent.clarifying_question} A: {reply}",
            ],
            "trace": _event("clarify", f"User answered: {reply}"),
        }

    def refuse_node(state: AgentState) -> dict[str, Any]:
        return {
            "answer": Answer(status="refused", text=REFUSAL, confidence="high"),
            "trace": _event("refuse", "Out of scope"),
        }

    def fast_plan_node(state: AgentState) -> dict[str, Any]:
        plan = fast_plan(state["intent"], request_text(state))
        assert plan is not None
        return {
            "contract": plan.contract,
            "plan": plan.steps,
            "plan_problems": [],
            "trace": _event("fast_plan", f"Single-tool plan: {plan.steps[0].tool}"),
        }

    async def planner_node(state: AgentState) -> dict[str, Any]:
        intent = state["intent"]
        catalog = toolbox.sql.compiler.model.catalog_text() if toolbox.sql else "(unavailable)"
        system = PLANNER_SYSTEM.format(
            today=date.today().isoformat(),
            user=_user_line(state["user"]),
            catalog=catalog,
            max_steps=settings.agent_max_plan_steps,
        )
        request = (
            f"Request: {request_text(state)}\nGoal: {intent.goal}\n"
            f"Requested output: {intent.output_format}\n"
            f"Available tools: {', '.join(sorted(toolbox.available()))}"
        )
        messages: list[Message] = [
            {"role": "system", "content": system},
            {"role": "user", "content": request},
        ]
        budget = state["budget"]
        problems = state.get("plan_problems") or []
        if problems and state.get("plan"):
            previous = Plan(contract=state["contract"], steps=state["plan"])
            messages += [
                {"role": "assistant", "content": previous.model_dump_json()},
                {"role": "user", "content": "Fix these problems:\n- " + "\n- ".join(problems)},
            ]
            budget = budget.model_copy(update={"replans": budget.replans + 1})
        with track_usage() as usage:
            plan = await llm.structured(messages, Plan, tier=ModelTier.STRONG)
        plan = ensure_chart(plan, intent)
        summary = "; ".join(f"{s.id} {s.tool}: {s.input}" for s in plan.steps)
        return {
            "contract": plan.contract,
            "plan": plan.steps,
            "budget": _charge(budget, usage),
            "trace": _event("planner", summary),
        }

    def plan_check_node(state: AgentState) -> dict[str, Any]:
        plan = Plan(contract=state["contract"], steps=state["plan"])
        problems = check_plan(plan, toolbox.available(), settings.agent_max_plan_steps)
        messages = [m for ms in problems.values() for m in ms]
        budget = state["budget"]
        if messages and budget.replans < budget.max_replans:
            return {"plan_problems": messages, "trace": _event("plan_check", "; ".join(messages))}
        steps, gaps = prune(plan, problems) if messages else (plan.steps, [])
        return {
            "plan": steps,
            "plan_problems": [],
            "runs": {s.id: StepRun(step_id=s.id) for s in steps},
            "evidence": {},
            "gaps": gaps,
            "trace": _event(
                "plan_check",
                f"{len(steps)} steps ready" + (f"; dropped: {'; '.join(messages)}" if messages
                                               else ""),
            ),
        }  # fmt: skip

    def route_plan(state: AgentState) -> Node:
        if state.get("plan_problems"):
            return "planner"
        return "executor" if state["plan"] else "synthesize"

    async def executor_node(state: AgentState) -> dict[str, Any]:
        runs = {k: v.model_copy() for k, v in state["runs"].items()}
        evidence: dict[str, Evidence] = dict(state.get("evidence") or {})
        budget = state["budget"]
        events: list[TraceEvent] = []

        for step in state["plan"]:
            run = runs[step.id]
            if run.status != "pending":
                continue
            blocked = [
                d
                for d in step.depends_on
                if runs[d].status in ("failed", "skipped")
                or (runs[d].status == "done" and not evidence[d].found)
            ]
            if blocked:
                run.status, run.error = (
                    "skipped",
                    f"needs {', '.join(blocked)}, which had no result",
                )
                events += _event("executor", f"{step.id} skipped: {run.error}")

        pending = [s for s in state["plan"] if runs[s.id].status == "pending"]
        reason = budget.exhausted()
        if reason:
            for step in pending:
                runs[step.id].status, runs[step.id].error = "skipped", f"not run: {reason}"
            events += _event("executor", f"Stopped: {reason}")
            return {"runs": runs, "trace": events}

        ready = [s for s in pending if all(runs[d].status == "done" for d in s.depends_on)]
        if not ready:
            for step in pending:
                runs[step.id].status, runs[step.id].error = "skipped", "dependencies never ran"
            return {"runs": runs, "trace": events}
        results = await asyncio.gather(
            *(
                execute_step(s, {d: evidence[d] for d in s.depends_on}, state["user"], toolbox)
                for s in ready
            )
        )
        for step, (ev, run, usage) in zip(ready, results, strict=True):
            runs[step.id] = run
            if ev is not None:
                evidence[step.id] = ev
            budget = _charge(budget, usage).model_copy(
                update={"tool_calls": budget.tool_calls + run.attempts}
            )
            outcome = (
                f"done{' (no result)' if ev is not None and not ev.found else ''}"
                if run.status == "done"
                else f"failed ({run.error_kind}): {run.error}"
            )
            fallback = f"; fallback: {run.fallback}" if run.fallback else ""
            events += _event(
                "executor", f"{step.id} {step.tool} {outcome}; attempts {run.attempts}{fallback}"
            )
        return {"runs": runs, "evidence": evidence, "budget": budget, "trace": events}

    def route_executor(state: AgentState) -> Node:
        if any(r.status == "pending" for r in state["runs"].values()):
            return "executor"
        return "synthesize"

    async def synthesize_node(state: AgentState) -> dict[str, Any]:
        evidence = state.get("evidence") or {}
        found = [ev for ev in evidence.values() if ev.found]
        if not found:
            return {"draft": None, "trace": _event("synthesize", "No evidence to answer from")}
        intent = state["intent"]
        runs = state.get("runs") or {}
        failures = [
            f"{sid}: {r.status} - {r.error}" for sid, r in runs.items() if r.status != "done"
        ] + [f"{ev.step_id}: no result - {ev.summary}" for ev in evidence.values() if not ev.found]
        assumptions = _assumptions(state)
        system = SYNTHESIS_SYSTEM.format(
            report_rules=REPORT_RULES if intent.output_format == "report" else "",
            goal=intent.goal,
            contract="\n".join(f"{c.id}: {c.description}" for c in state["contract"]),
            assumptions="\n".join(f"- {a}" for a in assumptions) or "(none)",
            evidence="\n\n".join(
                f'<evidence id="{ev.step_id}" tool="{ev.tool}">\n{ev.summary}\n</evidence>'
                for ev in found
            ),
            failures="\n".join(failures) or "(none)",
        )
        messages: list[Message] = [
            {"role": "system", "content": system},
            {"role": "user", "content": request_text(state)},
        ]
        with track_usage() as usage:
            draft = await llm.structured(messages, DraftAnswer, tier=ModelTier.STRONG)
        return {
            "draft": draft,
            "assumptions": assumptions,
            "budget": _charge(state["budget"], usage),
            "trace": _event("synthesize", f"Draft written (confidence {draft.confidence})"),
        }

    def render_node(state: AgentState) -> dict[str, Any]:
        draft = state.get("draft")
        evidence = state.get("evidence") or {}
        gaps = list(state.get("gaps") or [])
        gaps += [
            f"{r.step_id} ({_step_tool(state, r.step_id)}) {r.status}: {r.error}"
            for r in (state.get("runs") or {}).values()
            if r.status != "done"
        ]
        if draft is None:
            answer = Answer(
                status="failed",
                text="I could not find evidence to answer this question.",
                gaps=gaps or ["No step produced usable evidence."],
                assumptions=_assumptions(state),
                warnings=[w for ev in evidence.values() for w in ev.warnings],
                confidence="low",
            )
            return {"answer": answer, "trace": _event("render", "Failed: no evidence")}
        gaps = contract_gaps(state["contract"], draft, evidence) + gaps
        answer = build_answer(
            draft, evidence, gaps=gaps, assumptions=state.get("assumptions") or []
        )
        if state["intent"].output_format == "report":
            report = build_report(
                draft, answer, evidence, question=state["question"], user=state["user"]
            )
            links, problems = publish_report(report, settings.artifacts_dir)
            answer = answer.model_copy(
                update={"report": links, "warnings": answer.warnings + problems}
            )
        return {"answer": answer, "trace": _event("render", f"Answer {answer.status}")}

    graph = StateGraph(AgentState)
    graph.add_node("intent", intent_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("fast_plan", fast_plan_node)
    graph.add_node("planner", planner_node)
    graph.add_node("plan_check", plan_check_node)
    graph.add_node("executor", executor_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("render", render_node)

    graph.add_edge(START, "intent")
    graph.add_conditional_edges(
        "intent", route_intent, ["refuse", "clarify", "fast_plan", "planner"]
    )
    graph.add_edge("clarify", "intent")
    graph.add_edge("refuse", END)
    graph.add_edge("fast_plan", "plan_check")
    graph.add_edge("planner", "plan_check")
    graph.add_conditional_edges("plan_check", route_plan, ["planner", "executor", "synthesize"])
    graph.add_conditional_edges("executor", route_executor, ["executor", "synthesize"])
    graph.add_edge("synthesize", "render")
    graph.add_edge("render", END)
    return graph.compile(checkpointer=checkpointer)


def _step_tool(state: AgentState, step_id: str) -> str:
    return next((s.tool for s in state.get("plan") or [] if s.id == step_id), "?")


def _assumptions(state: AgentState) -> list[str]:
    found = [ev for ev in (state.get("evidence") or {}).values() if ev.found]
    return list(dict.fromkeys(a for ev in found for a in ev.assumptions))
