"""Run the agent graph per conversation thread: ask, pause for clarification, resume."""

import uuid
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel

from analytics_platform.agent.models import (
    AgentState,
    Answer,
    Budget,
    PlanStep,
    StepRun,
    TraceEvent,
)
from analytics_platform.security import UserContext

RECURSION_LIMIT = 60


class RunResult(BaseModel):
    thread_id: str
    status: Literal["answered", "needs_clarification"]
    answer: Answer | None = None
    clarifying_question: str | None = None
    plan: list[PlanStep] = []
    runs: dict[str, StepRun] = {}
    budget: Budget | None = None
    trace: list[TraceEvent] = []


class AgentService:
    def __init__(self, graph: CompiledStateGraph[AgentState, None, AgentState, AgentState]) -> None:
        self.graph = graph

    def _config(self, thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

    async def ask(
        self, question: str, user: UserContext, *, thread_id: str | None = None
    ) -> RunResult:
        thread_id = thread_id or uuid.uuid4().hex
        state: AgentState = {"question": question, "user": user, "clarifications": [], "trace": []}
        await self.graph.ainvoke(state, self._config(thread_id))
        return await self.result(thread_id)

    async def clarify(self, thread_id: str, reply: str) -> RunResult:
        await self.graph.ainvoke(Command(resume=reply), self._config(thread_id))
        return await self.result(thread_id)

    async def result(self, thread_id: str) -> RunResult:
        snapshot = await self.graph.aget_state(self._config(thread_id))
        values: dict[str, Any] = snapshot.values
        common: dict[str, Any] = {
            "thread_id": thread_id,
            "plan": values.get("plan") or [],
            "runs": values.get("runs") or {},
            "budget": values.get("budget"),
            "trace": values.get("trace") or [],
        }
        if snapshot.interrupts:
            payload = snapshot.interrupts[0].value
            return RunResult(
                status="needs_clarification", clarifying_question=payload["question"], **common
            )
        return RunResult(status="answered", answer=values.get("answer"), **common)
