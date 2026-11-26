from datetime import date

import pytest

from data_gen.timeline import Timeline, add_months, quarter_label


@pytest.mark.parametrize(
    ("as_of", "drop_q", "prev_q"),
    [
        (date(2026, 11, 26), date(2026, 7, 1), date(2026, 4, 1)),
        (date(2027, 2, 3), date(2026, 10, 1), date(2026, 7, 1)),
        (date(2026, 10, 1), date(2026, 7, 1), date(2026, 4, 1)),
    ],
)
def test_drop_quarter_is_last_complete_quarter(as_of: date, drop_q: date, prev_q: date) -> None:
    tl = Timeline.from_as_of(as_of)

    assert (tl.drop_q_start, tl.prev_q_start) == (drop_q, prev_q)
    assert tl.history_start == add_months(drop_q, -30)


def test_invoice_months_cover_history_through_as_of() -> None:
    tl = Timeline.from_as_of(date(2026, 11, 26))
    months = tl.invoice_months()

    assert months[0] == date(2024, 1, 1)
    assert months[-1] == date(2026, 11, 1)
    assert len(months) == 35


def test_add_months_and_labels() -> None:
    assert add_months(date(2026, 11, 15), 3) == date(2027, 2, 1)
    assert add_months(date(2026, 1, 31), -1) == date(2025, 12, 1)
    assert quarter_label(date(2026, 8, 9)) == "Q3 2026"
