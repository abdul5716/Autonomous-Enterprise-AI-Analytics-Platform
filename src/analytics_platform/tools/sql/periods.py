"""Resolve named time periods ("last quarter", "last 6 months") against the data's as-of date."""

from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, model_validator


def _month_start(d: date) -> date:
    return d.replace(day=1)


def _add_months(d: date, months: int) -> date:
    years, index = divmod(d.month - 1 + months, 12)
    return date(d.year + years, index + 1, 1)


def _quarter_start(d: date) -> date:
    return date(d.year, 3 * ((d.month - 1) // 3) + 1, 1)


PeriodName = Literal[
    "this_month",
    "last_month",
    "last_n_months",
    "this_quarter",
    "last_quarter",
    "last_n_quarters",
    "this_year",
    "last_year",
    "custom",
]


class TimeRange(BaseModel):
    period: PeriodName
    n: int | None = None
    start: date | None = None
    end: date | None = None  # exclusive

    @model_validator(mode="after")
    def _check(self) -> "TimeRange":
        if self.period in ("last_n_months", "last_n_quarters") and not (self.n and self.n > 0):
            raise ValueError(f"{self.period} needs a positive n")
        if self.period == "custom" and not (self.start and self.end and self.start < self.end):
            raise ValueError("custom period needs start < end")
        return self


class ResolvedPeriod(BaseModel):
    start: date
    end: date  # exclusive
    label: str
    months: int | None  # whole-month length, used to shift for comparisons
    partial: bool
    assumption: str | None = None

    def shifted(self, months: int) -> "ResolvedPeriod":
        start, end = _shift_day(self.start, months), _shift_day(self.end, months)
        return self.model_copy(
            update={
                "start": start,
                "end": end,
                "label": period_label(start, end),
                "partial": False,
                "assumption": None,
            }
        )


def _shift_day(d: date, months: int) -> date:
    """Same day-of-month `months` earlier, clamped to the end of the target month."""
    first = _add_months(d, -months)
    last_day = (_add_months(first, 1) - timedelta(days=1)).day
    return first.replace(day=min(d.day, last_day))


def _fmt_month(d: date) -> str:
    return f"{d:%b %Y}"


def _quarter_label(d: date) -> str:
    return f"{d.year}-Q{(d.month - 1) // 3 + 1}"


def period_label(start: date, end: date) -> str:
    """Human label for [start, end): a month, quarter, year, month range or date range."""
    if start.day != 1 or end.day != 1:
        return f"{start.isoformat()} to {(end - timedelta(days=1)).isoformat()}"
    months = (end.year - start.year) * 12 + end.month - start.month
    if months == 1:
        return _fmt_month(start)
    if months == 3 and start.month % 3 == 1:
        return _quarter_label(start)
    if months == 12 and start.month == 1:
        return str(start.year)
    if months % 3 == 0 and start.month % 3 == 1:
        return f"{_quarter_label(start)} to {_quarter_label(_add_months(end, -3))}"
    return f"{_fmt_month(start)} to {_fmt_month(_add_months(end, -1))}"


def resolve(tr: TimeRange, as_of: date) -> ResolvedPeriod:
    """Data is complete through `as_of`; ranges ending after it are flagged as partial."""
    this_month = _month_start(as_of)
    this_quarter = _quarter_start(as_of)
    n = tr.n or 1
    match tr.period:
        case "this_month":
            start, end = this_month, _add_months(this_month, 1)
        case "last_month":
            start, end = _add_months(this_month, -1), this_month
        case "last_n_months":
            start, end = _add_months(this_month, -n), this_month
        case "this_quarter":
            start, end = this_quarter, _add_months(this_quarter, 3)
        case "last_quarter":
            start, end = _add_months(this_quarter, -3), this_quarter
        case "last_n_quarters":
            start, end = _add_months(this_quarter, -3 * n), this_quarter
        case "this_year":
            start, end = date(as_of.year, 1, 1), date(as_of.year + 1, 1, 1)
        case "last_year":
            start, end = date(as_of.year - 1, 1, 1), date(as_of.year, 1, 1)
        case "custom":
            assert tr.start and tr.end
            start, end = tr.start, tr.end
    whole_months = start.day == 1 and end.day == 1
    months = (end.year - start.year) * 12 + end.month - start.month if whole_months else None
    label = period_label(start, end)
    if tr.period == "this_year":
        label = f"{as_of.year} year to date"
    if tr.period == "last_n_months":
        return ResolvedPeriod(
            start=start,
            end=end,
            label=label,
            months=months,
            partial=False,
            assumption=(
                f"Interpreted as the last {n} complete months ({label}); "
                f"{_fmt_month(this_month)} is excluded because it is not over yet."
            ),
        )
    partial = end > as_of + timedelta(days=1)
    assumption = (
        f"{label} is still in progress: data runs only through {as_of.isoformat()}."
        if partial
        else None
    )
    return ResolvedPeriod(
        start=start, end=end, label=label, months=months, partial=partial, assumption=assumption
    )


def previous_period(p: ResolvedPeriod) -> ResolvedPeriod:
    """The equally long period immediately before `p`."""
    if p.months is None:
        start = p.start - (p.end - p.start)
        return p.model_copy(
            update={
                "start": start,
                "end": p.start,
                "label": period_label(start, p.start),
                "partial": False,
                "assumption": None,
            }
        )
    return p.shifted(p.months)


def same_period_last_year(p: ResolvedPeriod) -> ResolvedPeriod:
    return p.shifted(12)
