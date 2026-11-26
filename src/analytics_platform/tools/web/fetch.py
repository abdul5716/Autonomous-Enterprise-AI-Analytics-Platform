"""Fetch pages safely (SSRF checks on every redirect, size cap) and extract the main text."""

import asyncio
from datetime import UTC, date, datetime
from typing import Protocol
from urllib.parse import urljoin

import httpx
import trafilatura

from analytics_platform.config import Settings
from analytics_platform.tools.base import ErrorKind, ToolError
from analytics_platform.tools.web.models import WebPage
from analytics_platform.tools.web.providers import FixtureSearch
from analytics_platform.tools.web.safety import check_url
from analytics_platform.tools.web.sanitize import sanitize_text, strip_hidden_html

USER_AGENT = "Mozilla/5.0 (compatible; NimbusResearchBot/0.1; +https://example.com/bot)"
MAX_REDIRECTS = 3


class PageFetcher(Protocol):
    async def fetch(self, url: str) -> WebPage: ...


def extract_page(url: str, html: str, max_chars: int) -> WebPage:
    """Main-content extraction (drops navigation, footers, cookie banners) plus sanitising."""
    visible, hidden = strip_hidden_html(html)
    text = trafilatura.extract(visible, include_comments=False, include_tables=True) or ""
    meta = trafilatura.extract_metadata(visible)
    if not text.strip():
        raise ToolError(f"No readable text at {url}", kind=ErrorKind.NOT_FOUND)
    sanitized = sanitize_text(text, max_chars)
    if sanitize_text(" ".join(hidden), max_chars).flags:
        sanitized.flags = sorted({*sanitized.flags, "hidden instructions"})
    published = None
    if meta and meta.date:
        try:
            published = date.fromisoformat(meta.date[:10])
        except ValueError:
            published = None
    return WebPage(
        url=url,
        title=(meta.title if meta and meta.title else url),
        text=sanitized.text,
        published=published,
        retrieved_at=datetime.now(UTC),
        injection_flags=sanitized.flags,
    )


class HttpFetcher:
    def __init__(
        self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.timeout_s = settings.web_fetch_timeout_s
        self.max_bytes = settings.web_max_page_bytes
        self.max_chars = settings.web_max_page_chars
        self.transport = transport

    async def fetch(self, url: str) -> WebPage:
        async with httpx.AsyncClient(
            timeout=self.timeout_s,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=False,
            transport=self.transport,
        ) as client:
            for _ in range(MAX_REDIRECTS + 1):
                await asyncio.to_thread(check_url, url)
                try:
                    html, location = await self._get(client, url)
                except httpx.TimeoutException as exc:
                    raise ToolError(f"Timed out fetching {url}", kind=ErrorKind.TIMEOUT) from exc
                except httpx.HTTPError as exc:
                    raise ToolError(
                        f"Could not fetch {url}: {exc}", kind=ErrorKind.UNAVAILABLE
                    ) from exc
                if location is None:
                    return await asyncio.to_thread(extract_page, url, html, self.max_chars)
                url = urljoin(url, location)
        raise ToolError(f"Too many redirects for {url}", kind=ErrorKind.UNAVAILABLE)

    async def _get(self, client: httpx.AsyncClient, url: str) -> tuple[str, str | None]:
        async with client.stream("GET", url) as resp:
            if resp.is_redirect:
                return "", resp.headers.get("location")
            if resp.status_code in (401, 403, 451):
                raise ToolError(
                    f"{url} refused access ({resp.status_code})", kind=ErrorKind.PERMISSION
                )
            if resp.status_code == 404:
                raise ToolError(f"{url} not found", kind=ErrorKind.NOT_FOUND)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise ToolError(f"{url} returned {resp.status_code}", kind=ErrorKind.TRANSIENT)
            content_type = resp.headers.get("content-type", "")
            if "html" not in content_type and "text" not in content_type:
                raise ToolError(
                    f"{url} is not a web page ({content_type})", kind=ErrorKind.INVALID_INPUT
                )
            body = bytearray()
            async for part in resp.aiter_bytes():
                body += part
                if len(body) > self.max_bytes:
                    break
            return body.decode(resp.encoding or "utf-8", errors="replace"), None


class FixtureFetcher:
    def __init__(self, fixtures: FixtureSearch, max_chars: int) -> None:
        self.fixtures = fixtures
        self.max_chars = max_chars

    async def fetch(self, url: str) -> WebPage:
        html = self.fixtures.html(url)
        if html is None:
            raise ToolError(f"{url} is not in the offline snapshot", kind=ErrorKind.NOT_FOUND)
        return extract_page(url, html, self.max_chars)


class RoutingFetcher:
    """Offline snapshot URLs are served locally; everything else goes over HTTP."""

    def __init__(self, fixture: FixtureFetcher, http: PageFetcher | None) -> None:
        self.fixture = fixture
        self.http = http

    async def fetch(self, url: str) -> WebPage:
        if url.split("/")[2].endswith(".example") or self.http is None:
            return await self.fixture.fetch(url)
        return await self.http.fetch(url)
