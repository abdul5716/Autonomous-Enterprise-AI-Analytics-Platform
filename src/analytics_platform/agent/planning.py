"""Deterministic planning: fixed plans for simple requests, and checks on LLM-written plans."""

from analytics_platform.agent.models import (
    TABLE_TOOLS,
    ContractItem,
    Intent,
    Plan,
    PlanStep,
)
from analytics_platform.tools.web.safety import egress_problems


def fast_plan(intent: Intent, request: str) -> Plan | None:
    """One-tool plans that need no planner call, or None if the request needs planning."""
    if intent.request_type == "document_qa":
        return Plan(
            contract=[ContractItem(id="C1", description=intent.goal)],
            steps=[PlanStep(id="S1", tool="documents", input=request, covers=["C1"])],
        )
    if intent.request_type == "data_query" and intent.output_format != "report":
        contract = [ContractItem(id="C1", description=intent.goal)]
        steps = [PlanStep(id="S1", tool="sql", input=request, covers=["C1"])]
        if intent.output_format == "chart":
            contract.append(ContractItem(id="C2", description="A chart of the result"))
            steps.append(
                PlanStep(id="S2", tool="chart", input=intent.goal, depends_on=["S1"], covers=["C2"])
            )
        return Plan(contract=contract, steps=steps)
    return None


def check_plan(plan: Plan, available: set[str], max_steps: int) -> dict[str, list[str]]:
    """Problems keyed by step id ("" for plan-level problems). Empty means the plan is valid."""
    problems: dict[str, list[str]] = {}

    def add(key: str, message: str) -> None:
        problems.setdefault(key, []).append(message)

    if len(plan.steps) > max_steps:
        add("", f"The plan has {len(plan.steps)} steps; the limit is {max_steps}.")
    contract_ids = {c.id for c in plan.contract}
    if len(contract_ids) != len(plan.contract):
        add("", "Contract item ids must be unique.")
    seen: dict[str, PlanStep] = {}
    for step in plan.steps:
        if step.id in seen:
            add(step.id, f"Duplicate step id {step.id}.")
            continue
        if step.tool not in available:
            add(step.id, f"{step.id} uses the {step.tool} tool, which is not available now.")
        if not step.input.strip():
            add(step.id, f"{step.id} has an empty input.")
        for dep in step.depends_on:
            if dep not in seen:
                add(step.id, f"{step.id} depends on {dep}, which is not an earlier step.")
        if step.tool in ("analysis", "chart") and not any(
            seen[d].tool in TABLE_TOOLS for d in step.depends_on if d in seen
        ):
            add(step.id, f"{step.id} ({step.tool}) must depend on an earlier sql or analysis step.")
        if step.tool == "web":
            for text in [step.input, *step.queries]:
                leaks = egress_problems(text)
                if leaks:
                    add(step.id, f"{step.id} web input {text!r} contains {', '.join(leaks)}.")
            if len(step.queries) > 3:
                add(step.id, f"{step.id} has more than 3 search queries.")
        unknown = [c for c in step.covers if c not in contract_ids]
        if unknown:
            add(step.id, f"{step.id} covers unknown contract items {unknown}.")
        seen[step.id] = step

    covered = {c for s in plan.steps for c in s.covers}
    for item in plan.contract:
        if item.id not in covered:
            add("", f"Contract item {item.id} ({item.description}) is not covered by any step.")
    return problems


def ensure_chart(plan: Plan, intent: Intent) -> Plan:
    """Add a chart step if a chart was asked for but the planner forgot it."""
    if intent.output_format != "chart" or any(s.tool == "chart" for s in plan.steps):
        return plan
    sources = [s for s in plan.steps if s.tool in TABLE_TOOLS]
    if not sources:
        return plan
    step_id = f"S{len(plan.steps) + 1}"
    while step_id in {s.id for s in plan.steps}:
        step_id += "b"
    chart = PlanStep(id=step_id, tool="chart", input=intent.goal, depends_on=[sources[-1].id])
    return plan.model_copy(update={"steps": [*plan.steps, chart]})


def prune(plan: Plan, problems: dict[str, list[str]]) -> tuple[list[PlanStep], list[str]]:
    """Drop invalid steps and everything that depends on them; return kept steps and gaps."""
    dropped = {sid for sid in problems if sid}
    changed = True
    while changed:
        changed = False
        for step in plan.steps:
            if step.id not in dropped and any(d in dropped for d in step.depends_on):
                dropped.add(step.id)
                changed = True
    kept = [s for s in plan.steps if s.id not in dropped]
    covered = {c for s in kept for c in s.covers}
    gaps = [
        f"Could not plan how to get: {item.description}"
        for item in plan.contract
        if item.id not in covered
    ]
    return kept, gaps
