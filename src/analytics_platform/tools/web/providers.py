"""Search providers, tried in order: Tavily (API key), ddgs engines, offline fixtures."""

import asyncio
import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import httpx

from analytics_platform.config import Settings
from analytics_platform.tools.base import ErrorKind, ToolError
from analytics_platform.tools.web.models import SearchHit

logger = logging.getLogger(__name__)

WORD = re.compile(r"[a-z0-9]+")


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, max_results: int) -> list[SearchHit]: ...


class TavilySearch:
    name = "tavily"
    URL = "https://api.tavily.com/search"

    def __init__(
        self, api_key: str, timeout_s: float, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._transport = transport

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        payload = {"query": query, "max_results": max_results, "search_depth": "basic"}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_s, transport=self._transport
            ) as client:
                resp = await client.post(self.URL, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ToolError("Tavily timed out", kind=ErrorKind.TIMEOUT) from exc
        except httpx.HTTPError as exc:
            raise ToolError(f"Tavily unreachable: {exc}", kind=ErrorKind.UNAVAILABLE) from exc
        if resp.status_code in (401, 403):
            raise ToolError("Tavily rejected the API key", kind=ErrorKind.PERMISSION)
        if resp.status_code == 429 or resp.status_code >= 500:
            raise ToolError(f"Tavily returned {resp.status_code}", kind=ErrorKind.TRANSIENT)
        resp.raise_for_status()
        return [
            SearchHit(
                title=r.get("title", ""),
                url=r["url"],
                snippet=r.get("content", ""),
                provider=self.name,
                published=_parse_date(r.get("published_date")),
            )
            for r in resp.json().get("results", [])
        ]


class DDGSSearch:
    """Free metasearch via the `ddgs` package. Some engines block cloud IPs, so try several."""

    name = "ddgs"

    def __init__(self, backends: list[str]) -> None:
        self.backends = backends

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        from ddgs import DDGS

        errors = []
        for backend in self.backends:
            try:
                results: list[dict[str, Any]] = await asyncio.to_thread(
                    DDGS().text, query, max_results=max_results, backend=backend
                )
            except Exception as exc:  # ddgs raises its own exceptions for blocks and empties
                errors.append(f"{backend}: {exc}")
                continue
            if results:
                return [
                    SearchHit(
                        title=r.get("title", ""),
                        url=r["href"],
                        snippet=r.get("body", ""),
                        provider=f"ddgs/{backend}",
                    )
                    for r in results
                    if r.get("href")
                ]
        raise ToolError(
            "No results from any ddgs engine: " + "; ".join(errors), kind=ErrorKind.UNAVAILABLE
        )


class FixtureSearch:
    """Keyword search over the offline web snapshot written by `make data`."""

    name = "fixtures"

    def __init__(self, fixtures_dir: Path) -> None:
        self.dir = fixtures_dir
        self._index: list[dict[str, Any]] | None = None
        self._texts: dict[str, str] = {}

    def index(self) -> list[dict[str, Any]]:
        if self._index is None:
            path = self.dir / "index.json"
            if not path.exists():
                raise ToolError(
                    f"Web fixtures missing at {path}; run `make data`", kind=ErrorKind.UNAVAILABLE
                )
            self._index = json.loads(path.read_text())
            for entry in self._index:
                self._texts[entry["url"]] = (self.dir / entry["file"]).read_text().lower()
        return self._index

    def html(self, url: str) -> str | None:
        entry = next((e for e in self.index() if e["url"] == url), None)
        return (self.dir / entry["file"]).read_text() if entry else None

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        words = {w for w in WORD.findall(query.lower()) if len(w) > 2}
        scored = []
        for entry in self.index():
            text = self._texts[entry["url"]]
            title = entry["title"].lower()
            score = sum(2 * (w in title) + (w in text) for w in words)
            if score:
                scored.append((score, entry))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [
            SearchHit(
                title=e["title"],
                url=e["url"],
                snippet=e["title"],
                provider=self.name,
                kind=e["kind"],
                published=_parse_date(e.get("published")),
            )
            for _, e in scored[:max_results]
        ]


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def build_providers(settings: Settings) -> list[SearchProvider]:
    fixtures = FixtureSearch(Path(settings.web_fixtures_dir))
    if settings.web_mode == "fixtures":
        return [fixtures]
    live: list[SearchProvider] = []
    if settings.tavily_api_key:
        live.append(
            TavilySearch(settings.tavily_api_key.get_secret_value(), settings.web_fetch_timeout_s)
        )
    live.append(DDGSSearch(settings.web_search_backends))
    return live if settings.web_mode == "live" else [*live, fixtures]
