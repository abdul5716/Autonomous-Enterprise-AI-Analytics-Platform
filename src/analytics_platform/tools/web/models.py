from datetime import date, datetime
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel

SourceKind = Literal["vendor", "press", "analyst", "reference", "forum", "blog", "unknown"]


class SearchHit(BaseModel):
    title: str
    url: str
    snippet: str = ""
    provider: str
    kind: SourceKind | None = None  # known for fixtures; inferred from the domain otherwise
    published: date | None = None


class WebPage(BaseModel):
    url: str
    title: str
    text: str
    published: date | None = None
    retrieved_at: datetime
    snippet_only: bool = False  # the fetch failed; only the search snippet is available
    injection_flags: list[str] = []

    @property
    def domain(self) -> str:
        host = urlsplit(self.url).hostname or ""
        return host.removeprefix("www.")


class ScoredSource(BaseModel):
    source_id: str  # W1, W2, ... within one research call
    page: WebPage
    kind: SourceKind
    credibility: float
    recency: float
    relevance: float
    score: float
    notes: list[str] = []
