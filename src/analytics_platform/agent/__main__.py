"""Ask the agent from the terminal.

uv run python -m analytics_platform.agent "Why did revenue drop last quarter?" --user finance.admin
"""

import argparse
import asyncio
import logging
from contextlib import AsyncExitStack

from analytics_platform.agent.checkpoint import memory_checkpointer, postgres_checkpointer
from analytics_platform.agent.graph import build_graph
from analytics_platform.agent.service import AgentService, RunResult
from analytics_platform.agent.steps import table_text
from analytics_platform.agent.toolbox import Toolbox
from analytics_platform.config import get_settings
from analytics_platform.llm import build_llm, track_usage
from analytics_platform.security import DEMO_USERS


def show(result: RunResult, *, verbose: bool) -> None:
    answer = result.answer
    if verbose:
        print("\n--- trace ---")
        for event in result.trace:
            print(f"[{event.node}] {event.message}")
    if answer is None:
        return
    print(f"\n=== {answer.status.upper()} (confidence {answer.confidence}) ===\n")
    print(answer.text)
    for table in answer.tables:
        print(f"\n{table.title}\n{table_text(table, limit=15)}")
    for chart in answer.charts:
        print(f"\n[chart] {chart.spec.title} ({chart.spec.type})")
    for label, items in (("Assumptions", answer.assumptions), ("Gaps", answer.gaps),
                         ("Warnings", answer.warnings)):  # fmt: skip
        if items:
            print(f"\n{label}:\n" + "\n".join(f"- {i}" for i in items))
    if answer.citations:
        print("\nSources:")
        for n, c in enumerate(answer.citations, start=1):
            print(f"[{n}] {c.title}" + (f" - {c.url}" if c.url else ""))
    if answer.report:
        print(f"\nReport: {answer.report.html}\n        {answer.report.pdf}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--user", default="finance.admin", choices=sorted(DEMO_USERS))
    parser.add_argument("--memory", action="store_true", help="checkpoint in memory, not Postgres")
    parser.add_argument("--quiet", action="store_true", help="hide the agent trace")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    settings = get_settings()
    llm = build_llm(settings)
    async with AsyncExitStack() as stack:
        checkpointer = (
            memory_checkpointer()
            if args.memory
            else await stack.enter_async_context(postgres_checkpointer(settings))
        )
        service = AgentService(build_graph(Toolbox.build(llm, settings), checkpointer))
        with track_usage() as usage:
            result = await service.ask(args.question, DEMO_USERS[args.user])
            while result.status == "needs_clarification":
                reply = await asyncio.to_thread(input, f"\n{result.clarifying_question}\n> ")
                result = await service.clarify(result.thread_id, reply)
        show(result, verbose=not args.quiet)
        print(
            f"\nLLM calls: {usage.calls} ({usage.cached_calls} cached), "
            f"cost ${usage.cost_usd:.4f}, thread {result.thread_id}"
        )


if __name__ == "__main__":
    asyncio.run(main())
