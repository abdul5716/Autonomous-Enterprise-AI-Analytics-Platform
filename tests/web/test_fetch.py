import httpx
import pytest

from analytics_platform.config import Settings
from analytics_platform.tools.base import ErrorKind, ToolError
from analytics_platform.tools.web.fetch import HttpFetcher
from analytics_platform.tools.web.providers import TavilySearch

PUBLIC = "http://93.184.216.34"
ARTICLE = (
    "<html><head><title>CRM news</title></head><body><nav>Home | About</nav><article>"
    "<h1>CRM news</h1><p>" + "Mid-market buyers compare CRM vendors on price and AI. " * 5 + "</p>"
    "</article><footer>Cookies</footer></body></html>"
)


def fetcher(handler: object, **overrides: object) -> HttpFetcher:
    settings = Settings(_env_file=None, **overrides)  # type: ignore[arg-type]
    return HttpFetcher(settings, transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


async def test_fetches_and_extracts_main_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html=ARTICLE)

    page = await fetcher(handler).fetch(f"{PUBLIC}/news")

    assert page.title == "CRM news"
    assert "compare CRM vendors" in page.text
    assert "Cookies" not in page.text


async def test_redirect_to_private_address_is_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest"})

    with pytest.raises(ToolError) as info:
        await fetcher(handler).fetch(f"{PUBLIC}/start")
    assert info.value.kind == ErrorKind.PERMISSION


@pytest.mark.parametrize(
    ("status", "kind"),
    [(403, ErrorKind.PERMISSION), (404, ErrorKind.NOT_FOUND), (503, ErrorKind.TRANSIENT)],
)
async def test_http_errors_are_classified(status: int, kind: ErrorKind) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    with pytest.raises(ToolError) as info:
        await fetcher(handler).fetch(f"{PUBLIC}/x")
    assert info.value.kind == kind


async def test_non_html_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"})

    with pytest.raises(ToolError, match="not a web page"):
        await fetcher(handler).fetch(f"{PUBLIC}/file.pdf")


async def test_large_pages_are_truncated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html=ARTICLE + "<p>" + "x " * 50_000 + "</p>")

    page = await fetcher(handler, web_max_page_bytes=5_000).fetch(f"{PUBLIC}/big")

    assert "compare CRM vendors" in page.text


async def test_tavily_results_and_errors() -> None:
    def ok(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer key"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "T",
                        "url": "https://a.com",
                        "content": "c",
                        "published_date": "2026-10-01T00:00:00Z",
                    }
                ]
            },
        )

    hits = await TavilySearch("key", 5, transport=httpx.MockTransport(ok)).search("q", 3)
    assert hits[0].url == "https://a.com" and str(hits[0].published) == "2026-10-01"

    def limited(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    with pytest.raises(ToolError) as info:
        await TavilySearch("key", 5, transport=httpx.MockTransport(limited)).search("q", 3)
    assert info.value.kind == ErrorKind.TRANSIENT
