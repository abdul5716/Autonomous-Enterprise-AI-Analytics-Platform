from datetime import UTC, date, datetime

import pytest

from analytics_platform.tools.web.models import WebPage
from analytics_platform.tools.web.scoring import (
    infer_kind,
    recency_score,
    relevance_score,
    score_source,
)

TODAY = date(2026, 11, 26)


def page(**kwargs: object) -> WebPage:
    values: dict[str, object] = {
        "url": "https://www.reuters.com/tech/crm",
        "title": "CRM market",
        "text": "Mid-market CRM buyers are price sensitive.",
        "published": date(2026, 10, 1),
        "retrieved_at": datetime(2026, 11, 26, tzinfo=UTC),
    }
    values.update(kwargs)
    return WebPage.model_validate(values)


@pytest.mark.parametrize(
    ("domain", "kind"),
    [
        ("hubspot.com", "vendor"),
        ("blog.hubspot.com", "vendor"),
        ("gartner.com", "analyst"),
        ("en.wikipedia.org", "reference"),
        ("reddit.com", "forum"),
        ("blog.random-site.io", "blog"),
        ("random-site.io", "unknown"),
    ],
)
def test_infer_kind(domain: str, kind: str) -> None:
    assert infer_kind(domain) == kind


@pytest.mark.parametrize(
    ("published", "score"),
    [(date(2026, 9, 1), 1.0), (date(2026, 1, 1), 0.7), (date(2020, 1, 1), 0.15), (None, 0.4)],
)
def test_recency(published: date | None, score: float) -> None:
    assert recency_score(published, TODAY) == score


def test_relevance_uses_best_matching_topic() -> None:
    p = page()

    assert relevance_score(["compare competitors", "mid-market CRM price"], p) == 1.0


def test_press_outranks_forum_and_injection_is_penalised() -> None:
    topics = ["CRM price"]
    press = score_source("W1", page(), None, topics, TODAY)
    forum = score_source("W2", page(url="https://reddit.com/r/crm"), None, topics, TODAY)
    attacked = score_source(
        "W3", page(injection_flags=["override instructions"]), None, topics, TODAY
    )

    assert press.score > forum.score
    assert attacked.credibility == press.credibility / 2
    assert "instructions aimed at AI" in attacked.notes[0]
