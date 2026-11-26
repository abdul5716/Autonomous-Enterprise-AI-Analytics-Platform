from datetime import date

import pytest

from data_gen.catalog import REFUND_APPROVAL_TOP, refund_approval_level
from data_gen.simulate import Dataset, revenue_by_region, simulate

AS_OF = date(2026, 11, 26)


@pytest.fixture(scope="module")
def ds() -> Dataset:
    return simulate(AS_OF, seed=42, n_customers=1200)


def test_planted_drop_is_about_fifteen_percent(ds: Dataset) -> None:
    by_region = revenue_by_region(ds)
    total_change = (by_region.net_drop.sum() / by_region.net_prev.sum() - 1) * 100

    assert -18 < total_change < -12


def test_emea_and_apac_drive_the_drop(ds: Dataset) -> None:
    by_region = revenue_by_region(ds)
    contribution = (by_region.net_drop - by_region.net_prev).sort_values()
    change = by_region.net_change_pct.to_dict()

    assert set(contribution.index[:2]) == {"EMEA", "APAC"}
    assert change["EMEA"] < -20
    assert change["APAC"] < -25
    assert -8 < change["NA"] < 2
    assert -3 < change["LATAM"] < 5


def test_emea_professional_price_rises_at_drop_quarter(ds: Dataset) -> None:
    tl = ds.timeline
    before = ds.prices.price("P02", "EMEA", tl.prev_q_start)
    after = ds.prices.price("P02", "EMEA", tl.drop_q_start)

    assert after == pytest.approx(before * 1.15, abs=0.01)
    assert ds.prices.price("P02", "NA", tl.drop_q_start) == ds.prices.price(
        "P02", "NA", tl.prev_q_start
    )


def test_no_data_after_as_of(ds: Dataset) -> None:
    assert ds.invoices.invoice_date.max() <= AS_OF
    assert ds.refunds.refund_date.max() <= AS_OF
    assert ds.support_tickets.created_at.max().date() <= AS_OF


def test_ended_subscriptions_are_not_invoiced_afterwards(ds: Dataset) -> None:
    ends = ds.subscriptions.set_index("subscription_id").end_date.dropna()
    invoiced = ds.invoices[ds.invoices.subscription_id.isin(ends.index)]

    assert (invoiced.invoice_date < invoiced.subscription_id.map(ends)).all()


def test_refund_approval_levels_match_policy(ds: Dataset) -> None:
    expected = ds.refunds.amount_usd.map(refund_approval_level)

    assert (ds.refunds.approval_level == expected).all()
    assert refund_approval_level(50_000.01) == REFUND_APPROVAL_TOP


def test_sla_credits_only_for_apac_non_smb(ds: Dataset) -> None:
    credits = ds.refunds[ds.refunds.reason == "sla_credit"]
    customers = ds.customers.set_index("customer_id")

    assert len(credits) > 0
    assert (credits.customer_id.map(customers.region) == "APAC").all()
    assert (credits.customer_id.map(customers.segment) != "SMB").all()


def test_same_seed_is_deterministic() -> None:
    a = simulate(AS_OF, seed=3, n_customers=200)
    b = simulate(AS_OF, seed=3, n_customers=200)

    assert a.invoices.amount_usd.sum() == b.invoices.amount_usd.sum()
    assert a.customers.company_name.tolist() == b.customers.company_name.tolist()
