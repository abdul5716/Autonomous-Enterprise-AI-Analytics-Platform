from datetime import date

import pytest

from analytics_platform.tools.sql.periods import (
    TimeRange,
    period_label,
    previous_period,
    resolve,
    same_period_last_year,
)

AS_OF = date(2026, 11, 26)


@pytest.mark.parametrize(
    ("period", "n", "start", "end", "label", "partial"),
    [
        ("last_quarter", None, date(2026, 7, 1), date(2026, 10, 1), "2026-Q3", False),
        ("this_quarter", None, date(2026, 10, 1), date(2027, 1, 1), "2026-Q4", True),
        ("last_n_months", 6, date(2026, 5, 1), date(2026, 11, 1), "May 2026 to Oct 2026", False),
        ("last_month", None, date(2026, 10, 1), date(2026, 11, 1), "Oct 2026", False),
        ("last_year", None, date(2025, 1, 1), date(2026, 1, 1), "2025", False),
        ("this_year", None, date(2026, 1, 1), date(2027, 1, 1), "2026 year to date", True),
        ("last_n_quarters", 4, date(2025, 10, 1), date(2026, 10, 1), "2025-Q4 to 2026-Q3", False),
    ],
)
def test_resolve_named_periods(
    period: str, n: int | None, start: date, end: date, label: str, partial: bool
) -> None:
    p = resolve(TimeRange.model_validate({"period": period, "n": n}), AS_OF)

    assert (p.start, p.end, p.label, p.partial) == (start, end, label, partial)


def test_last_n_months_explains_excluded_current_month() -> None:
    p = resolve(TimeRange(period="last_n_months", n=6), AS_OF)

    assert p.assumption is not None
    assert "Nov 2026 is excluded" in p.assumption


def test_partial_period_is_flagged() -> None:
    p = resolve(TimeRange(period="this_quarter"), AS_OF)

    assert p.assumption is not None and "in progress" in p.assumption


def test_previous_and_last_year_comparisons() -> None:
    q3 = resolve(TimeRange(period="last_quarter"), AS_OF)

    assert previous_period(q3).label == "2026-Q2"
    assert same_period_last_year(q3).label == "2025-Q3"


def test_previous_period_for_custom_days() -> None:
    p = resolve(TimeRange(period="custom", start=date(2026, 3, 10), end=date(2026, 3, 20)), AS_OF)
    prev = previous_period(p)

    assert (prev.start, prev.end) == (date(2026, 2, 28), date(2026, 3, 10))


def test_invalid_ranges_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive n"):
        TimeRange(period="last_n_months")
    with pytest.raises(ValueError, match="start < end"):
        TimeRange(period="custom", start=date(2026, 2, 1), end=date(2026, 1, 1))


def test_period_label_fallbacks() -> None:
    assert period_label(date(2026, 2, 1), date(2026, 4, 1)) == "Feb 2026 to Mar 2026"
    assert period_label(date(2026, 2, 3), date(2026, 2, 5)) == "2026-02-03 to 2026-02-04"
