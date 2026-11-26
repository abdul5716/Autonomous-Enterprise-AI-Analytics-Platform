"""The tools the agent can call, built once and shared by every request."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from analytics_platform.config import Settings
from analytics_platform.llm import BaseLLM
from analytics_platform.reliability import CircuitBreaker
from analytics_platform.tools.rag.tool import DocumentTool
from analytics_platform.tools.sandbox.tool import AnalysisTool
from analytics_platform.tools.sql.tool import SQLTool
from analytics_platform.tools.web.tool import WebResearchTool

logger = logging.getLogger(__name__)


@dataclass
class Toolbox:
    llm: BaseLLM
    settings: Settings
    sql: SQLTool | None = None
    documents: DocumentTool | None = None
    web: WebResearchTool | None = None
    analysis: AnalysisTool | None = None
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    def available(self) -> set[str]:
        tools = {"chart"}
        for name in ("sql", "documents", "web", "analysis"):
            if getattr(self, name) is not None:
                tools.add(name)
        return tools

    @classmethod
    def build(cls, llm: BaseLLM, settings: Settings) -> "Toolbox":
        """Build every tool that can be built; a tool that fails to start is left out and logged."""
        from analytics_platform.tools.rag.tool import build_document_tool
        from analytics_platform.tools.web.tool import build_web_tool

        return cls(
            llm=llm,
            settings=settings,
            sql=_attempt("sql", lambda: SQLTool(llm, settings)),
            documents=_attempt("documents", lambda: build_document_tool(llm, settings)),
            web=_attempt("web", lambda: build_web_tool(llm, settings)),
            analysis=_attempt("analysis", lambda: AnalysisTool(llm, settings)),
            breaker=CircuitBreaker(
                threshold=settings.agent_breaker_threshold,
                cooldown_s=settings.agent_breaker_cooldown_s,
            ),
        )


def _attempt[T](name: str, factory: Callable[[], T]) -> T | None:
    try:
        return factory()
    except Exception:
        logger.exception("Tool %s is unavailable", name)
        return None
