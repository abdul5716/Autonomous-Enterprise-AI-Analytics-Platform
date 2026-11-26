from dataclasses import dataclass
from datetime import date, timedelta


def month_start(d: date) -> date:
    return d.replace(day=1)


def add_months(d: date, months: int) -> date:
    """First day of the month `months` away from `d`'s month."""
    years, month_index = divmod(d.month - 1 + months, 12)
    return date(d.year + years, month_index + 1, 1)


def quarter_start(d: date) -> date:
    return date(d.year, 3 * ((d.month - 1) // 3) + 1, 1)


def quarter_label(d: date) -> str:
    return f"Q{(d.month - 1) // 3 + 1} {d.year}"


@dataclass(frozen=True)
class Timeline:
    """Dates relative to `as_of`, so "last quarter" always contains the planted drop."""

    as_of: date
    history_start: date
    prev_q_start: date
    drop_q_start: date
    current_q_start: date

    @classmethod
    def from_as_of(cls, as_of: date, history_quarters: int = 11) -> "Timeline":
        current_q = quarter_start(as_of)
        drop_q = add_months(current_q, -3)
        return cls(
            as_of=as_of,
            history_start=add_months(drop_q, -3 * (history_quarters - 1)),
            prev_q_start=add_months(drop_q, -3),
            drop_q_start=drop_q,
            current_q_start=current_q,
        )

    @property
    def drop_q_label(self) -> str:
        return quarter_label(self.drop_q_start)

    @property
    def prev_q_label(self) -> str:
        return quarter_label(self.prev_q_start)

    @property
    def general_price_increase(self) -> date:
        return add_months(self.history_start, 12)

    @property
    def ai_assistant_launch(self) -> date:
        return add_months(self.drop_q_start, -15)

    @property
    def emea_memo_date(self) -> date:
        return self.drop_q_start - timedelta(days=45)

    @property
    def incident_start(self) -> date:
        return self.drop_q_start + timedelta(days=9)

    @property
    def incident_end(self) -> date:
        return self.incident_start + timedelta(days=3)

    @property
    def orbit_promo_start(self) -> date:
        return self.drop_q_start - timedelta(days=12)

    def invoice_months(self) -> list[date]:
        months, m = [], self.history_start
        while m <= self.as_of:
            months.append(m)
            m = add_months(m, 1)
        return months

    def quarter_months(self, q_start: date) -> list[date]:
        return [add_months(q_start, i) for i in range(3)]
