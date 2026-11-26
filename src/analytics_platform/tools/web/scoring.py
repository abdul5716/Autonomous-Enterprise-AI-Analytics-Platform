"""Source quality: who published it, how recent it is, and how relevant it is to the question."""

import re
from datetime import date

from analytics_platform.tools.web.models import ScoredSource, SourceKind, WebPage

KIND_CREDIBILITY: dict[SourceKind, float] = {
    "vendor": 0.85,  # authoritative about itself, biased about competitors
    "analyst": 0.8,
    "press": 0.75,
    "reference": 0.6,
    "unknown": 0.4,
    "blog": 0.3,
    "forum": 0.25,
}

_DOMAINS: dict[str, SourceKind] = {
    "salesforce.com": "vendor",
    "hubspot.com": "vendor",
    "zoho.com": "vendor",
    "microsoft.com": "vendor",
    "oracle.com": "vendor",
    "gartner.com": "analyst",
    "forrester.com": "analyst",
    "idc.com": "analyst",
    "reuters.com": "press",
    "bloomberg.com": "press",
    "ft.com": "press",
    "wsj.com": "press",
    "techcrunch.com": "press",
    "theverge.com": "press",
    "zdnet.com": "press",
    "cnbc.com": "press",
    "wikipedia.org": "reference",
    "g2.com": "reference",
    "capterra.com": "reference",
    "reddit.com": "forum",
    "quora.com": "forum",
    "medium.com": "blog",
    "substack.com": "blog",
}

WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = {"the", "and", "for", "with", "what", "how", "are", "is", "of", "to", "in", "vs"}


def infer_kind(domain: str) -> SourceKind:
    for known, kind in _DOMAINS.items():
        if domain == known or domain.endswith("." + known):
            return kind
    if domain.startswith(("blog.", "community.", "forum.")):
        return "blog" if domain.startswith("blog.") else "forum"
    return "unknown"


def recency_score(published: date | None, today: date) -> float:
    if published is None:
        return 0.4
    age_days = (today - published).days
    if age_days < 0:
        return 0.4  # future-dated: suspicious
    if age_days <= 180:
        return 1.0
    if age_days <= 365:
        return 0.7
    if age_days <= 3 * 365:
        return 0.4
    return 0.15


def relevance_score(topics: list[str], page: WebPage) -> float:
    """Best share of a topic's words (the question or any search query) found on the page."""
    text = f"{page.title} {page.text}".lower()
    best = 0.0
    for topic in topics:
        words = {w for w in WORD.findall(topic.lower()) if w not in STOPWORDS and len(w) > 2}
        if words:
            best = max(best, sum(w in text for w in words) / len(words))
    return best


def score_source(
    source_id: str, page: WebPage, kind: SourceKind | None, topics: list[str], today: date
) -> ScoredSource:
    kind = kind or infer_kind(page.domain)
    credibility = KIND_CREDIBILITY[kind]
    notes = []
    if page.injection_flags:
        credibility *= 0.5
        notes.append("contained instructions aimed at AI (" + ", ".join(page.injection_flags) + ")")
    if page.snippet_only:
        credibility *= 0.7
        notes.append("page could not be fetched; search snippet only")
    if kind == "vendor":
        notes.append("vendor's own claims")
    recency = recency_score(page.published, today)
    relevance = relevance_score(topics, page)
    score = round(0.45 * credibility + 0.2 * recency + 0.35 * relevance, 3)
    return ScoredSource(
        source_id=source_id,
        page=page,
        kind=kind,
        credibility=round(credibility, 3),
        recency=recency,
        relevance=round(relevance, 3),
        score=score,
        notes=notes,
    )
