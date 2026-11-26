import json
import re
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import pytest

from analytics_platform.config import Settings
from analytics_platform.llm import FakeLLM, LLMError, Message
from analytics_platform.security import DEMO_USERS
from analytics_platform.tools.base import ErrorKind, ToolError
from analytics_platform.tools.web.cache import InMemoryTextCache
from analytics_platform.tools.web.fetch import FixtureFetcher, RoutingFetcher
from analytics_platform.tools.web.models import SearchHit
from analytics_platform.tools.web.providers import FixtureSearch, SearchProvider
from analytics_platform.tools.web.tool import WebResearchTool
from data_gen.timeline import Timeline
from data_gen.web_pages import write_web_fixtures

USER = DEMO_USERS["employee"]
ORBIT_QUOTE = "offering 50% off the first year and free data migration"


@pytest.fixture
def fixtures(tmp_path: Path) -> FixtureSearch:
    write_web_fixtures(Timeline.from_as_of(date.today()), tmp_path)
    return FixtureSearch(tmp_path)


def tool(
    fixtures: FixtureSearch,
    llm: FakeLLM,
    providers: Sequence[SearchProvider] | None = None,
    cache: InMemoryTextCache | None = None,
) -> WebResearchTool:
    return WebResearchTool(
        llm,
        list(providers or [fixtures]),
        RoutingFetcher(FixtureFetcher(fixtures, 6000), None),
        Settings(_env_file=None),
        cache=cache,
    )


def source_id(messages: Sequence[Message], title_part: str) -> str:
    match = re.search(
        rf'<source id="(W\d+)"[^>]*>\nTitle: [^\n]*{title_part}', messages[1]["content"]
    )
    assert match, f"no source titled {title_part!r}"
    return match.group(1)


def synthesis(
    quote: str = ORBIT_QUOTE, title: str = "Switch and Save"
) -> Callable[[Sequence[Message]], str]:
    def respond(messages: Sequence[Message]) -> str:
        return json.dumps(
            {
                "answerable": True,
                "summary": "Orbit CRM is discounting heavily in APAC.",
                "findings": [
                    {
                        "statement": "Orbit offers 50% off",
                        "source_ids": [source_id(messages, title)],
                        "quote": quote,
                    }
                ],
            }
        )

    return respond


async def test_research_cites_web_sources_with_urls(fixtures: FixtureSearch) -> None:
    llm = FakeLLM().when("Sources:", synthesis())

    r = await tool(fixtures, llm).research("Orbit CRM promotion", USER, queries=["Orbit CRM APAC"])

    assert r.answered and r.providers_used == ["fixtures"]
    assert r.citations[0].source_type == "web"
    assert r.citations[0].url == "https://www.orbitcrm.example/press/switch-and-save-apac"
    assert "vendor's own claims" in (r.citations[0].detail or "")


async def test_queries_are_planned_by_the_llm(fixtures: FixtureSearch) -> None:
    llm = (
        FakeLLM()
        .when("Write up to 3 web search queries", {"queries": ["Orbit CRM Switch and Save"]})
        .when("Sources:", synthesis())
    )

    r = await tool(fixtures, llm).research("What is Orbit doing in Asia?", USER)

    assert r.queries == ["Orbit CRM Switch and Save"]


async def test_query_planning_failure_falls_back_to_the_question(fixtures: FixtureSearch) -> None:
    def broken(messages: Sequence[Message]) -> str:
        raise LLMError("provider down")

    llm = FakeLLM().when("Write up to 3", broken).when("Sources:", synthesis())

    r = await tool(fixtures, llm).research("Orbit CRM Switch and Save APAC", USER)

    assert r.queries == ["Orbit CRM Switch and Save APAC"] and r.answered


async def test_queries_with_internal_data_are_never_sent(fixtures: FixtureSearch) -> None:
    llm = FakeLLM()

    r = await tool(fixtures, llm).research(
        "why", USER, queries=["EMEA revenue fell to $10.7M", "POL-FIN-007 refunds"]
    )

    assert not r.answered and llm.calls == []
    assert "money amount" in r.warnings[0] and "internal document id" in r.warnings[1]


class FailingProvider:
    name = "live"

    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.calls += 1
        raise ToolError("rate limited", kind=ErrorKind.TRANSIENT)


async def test_falls_back_to_next_provider(fixtures: FixtureSearch) -> None:
    llm = FakeLLM().when("Sources:", synthesis())
    live = FailingProvider()

    r = await tool(fixtures, llm, providers=[live, fixtures]).research(
        "Orbit", USER, queries=["Orbit CRM APAC"]
    )

    assert r.answered and r.providers_used == ["fixtures"]
    assert "Search provider live failed for 'Orbit CRM APAC' (transient)" in r.warnings[0]


async def test_search_results_and_pages_are_cached(fixtures: FixtureSearch) -> None:
    cache = InMemoryTextCache()
    llm = FakeLLM().when("Sources:", synthesis())
    t = tool(fixtures, llm, cache=cache)

    await t.research("Orbit", USER, queries=["Orbit CRM APAC"])
    keys = set(cache.store)
    fixtures.search = None  # type: ignore[assignment,method-assign]
    second = await t.research("Orbit", USER, queries=["Orbit CRM APAC"])

    assert second.answered and set(cache.store) == keys
    assert any(":search:fixtures:" in k for k in keys) and any(":page:" in k for k in keys)


async def test_injection_page_is_flagged_and_downranked(fixtures: FixtureSearch) -> None:
    def respond(messages: Sequence[Message]) -> str:
        content = messages[1]["content"]
        assert "ignore all previous instructions" not in content
        assert "developer mode" not in content
        return synthesis("HubSpot is easiest to start with", "Top 10 CRMs")(messages)

    llm = FakeLLM().when("Sources:", respond)

    r = await tool(fixtures, llm).research(
        "best CRM reviews", USER, queries=["Top 10 CRMs honest reviews"]
    )

    blog = next(s for s in r.sources if s.page.domain == "best-crm-reviews.example")
    assert "hidden instructions" in blog.page.injection_flags
    assert any("Removed AI-directed instructions" in w for w in r.warnings)


async def test_misquoted_findings_are_repaired(fixtures: FixtureSearch) -> None:
    llm = (
        FakeLLM()
        .when("Fix these citation problems", synthesis())
        .when("Sources:", synthesis("Orbit is giving everything away for free"))
    )

    r = await tool(fixtures, llm).research("Orbit", USER, queries=["Orbit CRM APAC"])

    assert r.answered and len(llm.calls) == 2


async def test_unfetchable_pages_fall_back_to_snippets(fixtures: FixtureSearch) -> None:
    class SnippetProvider:
        name = "snippets"

        async def search(self, query: str, max_results: int) -> list[SearchHit]:
            return [
                SearchHit(
                    title="Gone",
                    url="https://gone.example/x",
                    snippet="Orbit CRM offers 50% off in APAC.",
                    provider=self.name,
                    kind="press",
                    published=date.today(),
                )
            ]

    def respond(messages: Sequence[Message]) -> str:
        return synthesis("Orbit CRM offers 50% off in APAC", "Gone")(messages)

    llm = FakeLLM().when("Sources:", respond)

    r = await tool(fixtures, llm, providers=[SnippetProvider()]).research(
        "Orbit CRM offer", USER, queries=["Orbit CRM offer"]
    )

    assert r.sources[0].page.snippet_only
    assert "search snippet only" in (r.citations[0].detail or "")


def test_snapshot_pages_parse(fixtures: FixtureSearch) -> None:
    for entry in fixtures.index():
        html = fixtures.html(entry["url"])
        assert html is not None and entry["title"] in html
