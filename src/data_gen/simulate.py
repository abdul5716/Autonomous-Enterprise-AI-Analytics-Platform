"""Simulate Nimbus Analytics' business history, including the planted revenue drop."""

import itertools
import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker

from data_gen.catalog import (
    ADDON_ADOPTION,
    COMPETITORS,
    COUNTRIES,
    EMEA_PROFESSIONAL_INCREASE,
    FICTIONAL_COMPETITOR,
    GENERAL_PRICE_INCREASE,
    INDUSTRIES,
    PRODUCTS,
    PRODUCTS_BY_ID,
    REGION_CUSTOMER_SHARE,
    REGION_PRICE_FACTOR,
    REGION_QOQ_TARGET,
    REGIONS,
    SEGMENTS,
    SLA_CREDIT_RANGE,
    refund_approval_level,
)
from data_gen.timeline import Timeline, add_months, month_start

CORE_PRODUCTS = {"P01", "P02", "P03"}
REP_COUNTS = {"NA": 16, "EMEA": 12, "APAC": 7, "LATAM": 5}
SCENARIO_CHURN_SHARE = {"NA": 0.8, "EMEA": 0.6, "APAC": 0.75, "LATAM": 0.8}
BASELINE_CANCEL_REASONS = ["budget_cuts", "switched_vendor", "business_closed", "low_usage"]
BASELINE_CANCEL_WEIGHTS = [0.30, 0.30, 0.15, 0.25]
LOST_REASONS = ["price", "competitor", "no_decision", "timing"]

TICKET_SUBJECTS = {
    "how_to": [
        "How do I configure pipeline stages?",
        "Help exporting reports to Excel",
        "Setting up user roles and permissions",
        "Importing contacts from CSV",
    ],
    "bug": [
        "Dashboard widgets not refreshing",
        "Email sync creates duplicate contacts",
        "Search returns incomplete results",
        "Mobile app crashes on login",
    ],
    "billing": [
        "Invoice amount looks incorrect",
        "Request to update billing contact",
        "Question about seat count on invoice",
        "Need a copy of last invoice",
    ],
    "outage": ["Application unavailable", "Very slow page loads", "API returning 503 errors"],
}
INCIDENT_SUBJECTS = [
    "Cannot log in - APAC data center",
    "Dashboards timing out since this morning",
    "API returning 503 errors for all requests",
    "Data sync stopped, records missing",
]
PRICE_SUBJECTS = [
    "Question about Professional plan price increase",
    "Requesting discount after price change",
    "Considering downgrade due to new pricing",
]


@dataclass
class Customer:
    customer_id: str
    company_name: str
    industry: str
    segment: str
    country_code: str
    region: str
    signup_date: date
    owner_rep_id: str


@dataclass
class Subscription:
    subscription_id: str
    customer_id: str
    product_id: str
    seats: int
    discount_pct: float
    start_date: date
    end_date: date | None = None
    cancel_reason: str | None = None
    competitor: str | None = None

    def first_invoice_month(self) -> date:
        return self.start_date if self.start_date.day == 1 else add_months(self.start_date, 1)

    def is_active_on(self, day: date) -> bool:
        return self.start_date <= day and (self.end_date is None or self.end_date > day)


@dataclass
class DropSummary:
    churned: dict[str, int] = field(default_factory=lambda: dict.fromkeys(REGIONS, 0))
    downgraded: dict[str, int] = field(default_factory=lambda: dict.fromkeys(REGIONS, 0))


@dataclass
class Dataset:
    timeline: Timeline
    prices: "PriceBook"
    drop_summary: DropSummary
    countries: pd.DataFrame
    sales_reps: pd.DataFrame
    customers: pd.DataFrame
    products: pd.DataFrame
    price_history: pd.DataFrame
    subscriptions: pd.DataFrame
    opportunities: pd.DataFrame
    support_tickets: pd.DataFrame
    invoices: pd.DataFrame
    refunds: pd.DataFrame
    budgets: pd.DataFrame


class PriceBook:
    """Per-seat monthly list prices by product and region, with their change history."""

    def __init__(self, tl: Timeline) -> None:
        self._history: dict[tuple[str, str], list[tuple[date, float]]] = {}
        earliest = date(tl.history_start.year - 4, 1, 1)
        for product in PRODUCTS:
            launch = tl.ai_assistant_launch if product.product_id == "P05" else earliest
            for region, factor in REGION_PRICE_FACTOR.items():
                base = round(product.base_price * factor, 2)
                points = [(launch, base)]
                increase_at = max(tl.general_price_increase, launch)
                if increase_at > launch:
                    points.append((increase_at, round(base * (1 + GENERAL_PRICE_INCREASE), 2)))
                if product.product_id == "P02" and region == "EMEA":
                    new_price = round(points[-1][1] * (1 + EMEA_PROFESSIONAL_INCREASE), 2)
                    points.append((tl.drop_q_start, new_price))
                self._history[(product.product_id, region)] = points

    def price(self, product_id: str, region: str, day: date) -> float:
        points = self._history[(product_id, region)]
        return next(p for start, p in reversed(points) if start <= day)

    def history_rows(self) -> list[dict[str, Any]]:
        rows = []
        for (product_id, region), points in self._history.items():
            for i, (start, price) in enumerate(points):
                end = points[i + 1][0] if i + 1 < len(points) else None
                rows.append(
                    {
                        "product_id": product_id,
                        "region": region,
                        "unit_price_usd": price,
                        "effective_from": start,
                        "effective_to": end,
                    }
                )
        return rows


def _ids(prefix: str, width: int = 6) -> Iterator[str]:
    return (f"{prefix}{n:0{width}d}" for n in itertools.count(1))


def _pick[T](rng: np.random.Generator, items: Sequence[T], p: Sequence[float] | None = None) -> T:
    weights = None if p is None else np.asarray(p, dtype=float) / np.sum(p)
    return items[int(rng.choice(len(items), p=weights))]


def _days(rng: np.random.Generator, low: int, high: int) -> timedelta:
    return timedelta(days=int(rng.integers(low, high)))


def _log_uniform_int(rng: np.random.Generator, low: int, high: int) -> int:
    return round(math.exp(rng.uniform(math.log(low), math.log(high))))


def simulate(as_of: date, *, seed: int = 42, n_customers: int = 2000) -> Dataset:
    rng = np.random.default_rng(seed)
    fake = Faker("en_US")
    fake.seed_instance(seed)
    tl = Timeline.from_as_of(as_of)
    prices = PriceBook(tl)
    sub_ids = _ids("S")

    reps = _make_reps(rng, fake)
    customers = _make_customers(rng, fake, tl, reps, n_customers)
    subs = _make_subscriptions(rng, tl, customers, sub_ids)

    baseline_invoices = _invoice_records(subs, customers, prices, tl)
    budgets = _budgets(baseline_invoices)
    drop_summary = _plant_revenue_drop(rng, tl, customers, subs, baseline_invoices, sub_ids)

    invoices = _invoice_records(subs, customers, prices, tl)
    _assign_invoice_ids_and_status(rng, tl, invoices)
    refunds = _refunds(rng, tl, invoices, customers)
    opportunities = _opportunities(rng, fake, tl, prices, reps)
    tickets = _tickets(rng, tl, customers, subs)

    return Dataset(
        timeline=tl,
        prices=prices,
        drop_summary=drop_summary,
        countries=pd.DataFrame(
            [
                {"country_code": code, "country_name": name, "region": region}
                for region, countries in COUNTRIES.items()
                for code, name in countries
            ]
        ),
        sales_reps=pd.DataFrame(reps),
        customers=_customer_frame(customers, subs),
        products=pd.DataFrame(
            [
                {
                    "product_id": p.product_id,
                    "product_name": p.name,
                    "product_line": p.line,
                    "edition": p.edition,
                    "launched_date": tl.ai_assistant_launch if p.product_id == "P05" else None,
                }
                for p in PRODUCTS
            ]
        ),
        price_history=pd.DataFrame(prices.history_rows()),
        subscriptions=pd.DataFrame([asdict(s) for s in subs]),
        opportunities=pd.DataFrame(opportunities),
        support_tickets=pd.DataFrame(tickets).astype({"csat_score": "Int64"}),
        invoices=pd.DataFrame(invoices),
        refunds=pd.DataFrame(refunds),
        budgets=pd.DataFrame(budgets),
    )


def _make_reps(rng: np.random.Generator, fake: Faker) -> list[dict[str, Any]]:
    reps, ids = [], _ids("R", 3)
    for region, count in REP_COUNTS.items():
        for _ in range(count):
            first, last = fake.first_name(), fake.last_name()
            reps.append(
                {
                    "rep_id": next(ids),
                    "full_name": f"{first} {last}",
                    "email": f"{first}.{last}@nimbus-analytics.example".lower(),
                    "region": region,
                    "hired_date": date(2019, 1, 1) + _days(rng, 0, 5 * 365),
                }
            )
    return reps


def _make_customers(
    rng: np.random.Generator,
    fake: Faker,
    tl: Timeline,
    reps: list[dict[str, Any]],
    n: int,
) -> dict[str, Customer]:
    regions = list(REGION_CUSTOMER_SHARE)
    segments = list(SEGMENTS)
    reps_by_region = defaultdict(list)
    for rep in reps:
        reps_by_region[rep["region"]].append(rep["rep_id"])

    customers: dict[str, Customer] = {}
    names: set[str] = set()
    ids = _ids("C")
    new_span = (tl.as_of - timedelta(days=7) - tl.history_start).days
    for _ in range(n):
        region = _pick(rng, regions, list(REGION_CUSTOMER_SHARE.values()))
        name = fake.company()
        while name in names:
            name = f"{fake.company()} {fake.company_suffix()}"
        names.add(name)
        if rng.random() < 0.5:
            signup = tl.history_start - _days(rng, 30, 3 * 365)
        else:
            signup = tl.history_start + _days(rng, 0, new_span)
        customer = Customer(
            customer_id=next(ids),
            company_name=name,
            industry=_pick(rng, INDUSTRIES),
            segment=_pick(rng, segments, [s.share for s in SEGMENTS.values()]),
            country_code=_pick(rng, COUNTRIES[region])[0],
            region=region,
            signup_date=signup,
            owner_rep_id=_pick(rng, reps_by_region[region]),
        )
        customers[customer.customer_id] = customer
    return customers


def _make_subscriptions(
    rng: np.random.Generator,
    tl: Timeline,
    customers: dict[str, Customer],
    sub_ids: Iterator[str],
) -> list[Subscription]:
    subs: list[Subscription] = []
    for c in customers.values():
        seg = SEGMENTS[c.segment]
        seats = _log_uniform_int(rng, *seg.seats)
        discount = round(float(rng.uniform(*seg.discount)), 3)
        core_ids = [pid for pid, _ in seg.core_products]
        core = _pick(rng, core_ids, [w for _, w in seg.core_products])
        own = [Subscription(next(sub_ids), c.customer_id, core, seats, discount, c.signup_date)]
        for addon, adoption in ADDON_ADOPTION.items():
            if rng.random() >= adoption:
                continue
            start = c.signup_date + _days(rng, 0, 365)
            if addon == "P05":
                start = max(start, tl.ai_assistant_launch + _days(rng, 0, 90))
            if start < tl.as_of:
                addon_seats = max(1, round(seats * rng.uniform(0.3, 1.0)))
                own.append(
                    Subscription(next(sub_ids), c.customer_id, addon, addon_seats, discount, start)
                )

        churn_date = _baseline_churn_date(rng, tl, own[0], seg.monthly_churn)
        if churn_date is not None:
            reason = _pick(rng, BASELINE_CANCEL_REASONS, BASELINE_CANCEL_WEIGHTS)
            competitor = (
                _baseline_competitor(rng, c.region) if reason == "switched_vendor" else None
            )
            own = [s for s in own if s.start_date < churn_date]
            for s in own:
                s.end_date, s.cancel_reason, s.competitor = churn_date, reason, competitor
        subs.extend(own)
    return subs


def _baseline_churn_date(
    rng: np.random.Generator, tl: Timeline, core: Subscription, monthly_churn: float
) -> date | None:
    m = max(core.first_invoice_month(), tl.history_start)
    while m <= tl.as_of:
        if rng.random() < monthly_churn:
            churn = m + _days(rng, 1, 28)
            return churn if churn <= tl.as_of else None
        m = add_months(m, 1)
    return None


def _baseline_competitor(rng: np.random.Generator, region: str) -> str:
    orbit_weight = 0.10 if region == "APAC" else 0.02
    return _pick(rng, COMPETITORS, [0.35, 0.35, 0.20, orbit_weight])


def _invoice_records(
    subs: list[Subscription],
    customers: dict[str, Customer],
    prices: PriceBook,
    tl: Timeline,
) -> list[dict[str, Any]]:
    months = tl.invoice_months()
    records = []
    for s in subs:
        region = customers[s.customer_id].region
        first = s.first_invoice_month()
        for m in months:
            if m < first:
                continue
            if s.end_date is not None and s.end_date <= m:
                break
            unit = prices.price(s.product_id, region, m)
            records.append(
                {
                    "subscription_id": s.subscription_id,
                    "customer_id": s.customer_id,
                    "product_id": s.product_id,
                    "region": region,
                    "invoice_date": m,
                    "seats": s.seats,
                    "unit_price_usd": unit,
                    "discount_pct": s.discount_pct,
                    "amount_usd": round(s.seats * unit * (1 - s.discount_pct), 2),
                }
            )
    return records


def _assign_invoice_ids_and_status(
    rng: np.random.Generator, tl: Timeline, invoices: list[dict[str, Any]]
) -> None:
    invoices.sort(key=lambda r: (r["invoice_date"], r["customer_id"], r["subscription_id"]))
    ids = _ids("INV-", 7)
    recent = tl.as_of - timedelta(days=45)
    for inv in invoices:
        inv["invoice_id"] = next(ids)
        if inv["invoice_date"] < recent:
            inv["status"] = "overdue" if rng.random() < 0.01 else "paid"
        else:
            inv["status"] = _pick(rng, ["paid", "open"], [0.6, 0.4])


def _budgets(baseline_invoices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Budget = pre-drop plan: baseline revenue plus 4%, by month, region and product line."""
    totals: dict[tuple[date, str, str], float] = defaultdict(float)
    for inv in baseline_invoices:
        line = PRODUCTS_BY_ID[inv["product_id"]].line
        totals[(inv["invoice_date"], inv["region"], line)] += inv["amount_usd"]
    return [
        {
            "budget_month": month,
            "region": region,
            "product_line": line,
            "amount_usd": round(total * 1.04, -2),
        }
        for (month, region, line), total in sorted(totals.items())
    ]


def _plant_revenue_drop(
    rng: np.random.Generator,
    tl: Timeline,
    customers: dict[str, Customer],
    subs: list[Subscription],
    baseline_invoices: list[dict[str, Any]],
    sub_ids: Iterator[str],
) -> DropSummary:
    """Churn or downgrade customers until each region hits its quarter-over-quarter target."""
    drop_months = tl.quarter_months(tl.drop_q_start)
    prev_months = tl.quarter_months(tl.prev_q_start)
    revenue: dict[tuple[str, date], float] = defaultdict(float)
    for inv in baseline_invoices:
        revenue[(inv["customer_id"], inv["invoice_date"])] += inv["amount_usd"]

    subs_by_customer: dict[str, list[Subscription]] = defaultdict(list)
    for s in subs:
        subs_by_customer[s.customer_id].append(s)

    summary = DropSummary()
    safe_until = tl.drop_q_start + timedelta(days=31)
    for region, target in REGION_QOQ_TARGET.items():
        members = [c for c in customers.values() if c.region == region]
        prev = sum(revenue[(c.customer_id, m)] for c in members for m in prev_months)
        cur = sum(revenue[(c.customer_id, m)] for c in members for m in drop_months)
        needed = cur - prev * (1 + target)
        if needed <= 0:
            continue

        candidates = [
            c
            for c in members
            if c.signup_date < tl.prev_q_start
            and all(
                s.end_date is None or s.end_date > safe_until
                for s in subs_by_customer[c.customer_id]
            )
        ]
        weights = [_scenario_weight(region, c, subs_by_customer[c.customer_id]) for c in candidates]
        order = rng.choice(
            len(candidates), size=len(candidates), replace=False, p=np.array(weights) / sum(weights)
        )

        cut, tolerance = 0.0, 0.005 * prev
        for idx in order:
            if cut >= needed:
                break
            c = candidates[int(idx)]
            q_revenue = {m: revenue[(c.customer_id, m)] for m in drop_months}
            own = subs_by_customer[c.customer_id]
            churn = rng.random() < SCENARIO_CHURN_SHARE[region]
            churn_date = tl.drop_q_start + timedelta(days=int(rng.integers(-20, 21)))
            removed_share = float(rng.uniform(0.4, 0.6))
            if churn:
                loss = sum(v for m, v in q_revenue.items() if m >= churn_date)
            else:
                loss = sum(q_revenue.values()) * removed_share
            if cut + loss > needed + tolerance:
                continue

            cut += loss
            if churn:
                reason, competitor = _scenario_reason(rng, region)
                for s in own:
                    if s.end_date is None or s.end_date > churn_date:
                        s.end_date, s.cancel_reason, s.competitor = churn_date, reason, competitor
                summary.churned[region] += 1
            else:
                for s in [s for s in own if s.is_active_on(tl.drop_q_start)]:
                    replacement = Subscription(
                        next(sub_ids),
                        s.customer_id,
                        s.product_id,
                        max(1, round(s.seats * (1 - removed_share))),
                        s.discount_pct,
                        tl.drop_q_start,
                        end_date=s.end_date,
                        cancel_reason=s.cancel_reason,
                        competitor=s.competitor,
                    )
                    s.end_date, s.cancel_reason, s.competitor = tl.drop_q_start, "downgrade", None
                    subs.append(replacement)
                summary.downgraded[region] += 1
    return summary


def _scenario_weight(region: str, customer: Customer, subs: list[Subscription]) -> float:
    if region == "EMEA":
        return 4.0 if any(s.product_id == "P02" for s in subs) else 1.0
    if region == "APAC":
        return 2.0 if customer.segment != "SMB" else 1.0
    return 1.0


def _scenario_reason(rng: np.random.Generator, region: str) -> tuple[str, str | None]:
    if region == "EMEA":
        if rng.random() < 0.7:
            return "price_increase", None
        return "switched_vendor", _pick(rng, ["Salesforce", "HubSpot", "Zoho"])
    if region == "APAC":
        if rng.random() < 0.55:
            return "switched_vendor", FICTIONAL_COMPETITOR
        return "service_reliability", None
    reason = _pick(rng, BASELINE_CANCEL_REASONS, BASELINE_CANCEL_WEIGHTS)
    return reason, _baseline_competitor(rng, region) if reason == "switched_vendor" else None


def _refunds(
    rng: np.random.Generator,
    tl: Timeline,
    invoices: list[dict[str, Any]],
    customers: dict[str, Customer],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def add(inv: dict[str, Any], amount: float, refund_date: date, reason: str) -> None:
        records.append(
            {
                "invoice_id": inv["invoice_id"],
                "customer_id": inv["customer_id"],
                "refund_date": refund_date,
                "amount_usd": amount,
                "reason": reason,
                "approval_level": refund_approval_level(amount),
            }
        )

    draws = rng.random(len(invoices))
    for inv, draw in zip(invoices, draws, strict=True):
        if draw >= 0.02:
            continue
        refund_date = inv["invoice_date"] + _days(rng, 3, 46)
        if refund_date > tl.as_of:
            continue
        share = _pick(rng, [0.1, 0.25, 0.5, 1.0], [0.4, 0.3, 0.2, 0.1])
        reason = _pick(
            rng, ["billing_error", "goodwill_credit", "prorated_cancellation"], [0.4, 0.35, 0.25]
        )
        add(inv, round(inv["amount_usd"] * share, 2), refund_date, reason)

    incident_month = month_start(tl.incident_start)
    for inv in invoices:
        if (
            inv["invoice_date"] == incident_month
            and inv["region"] == "APAC"
            and inv["product_id"] in CORE_PRODUCTS
            and customers[inv["customer_id"]].segment != "SMB"
        ):
            credit = round(inv["amount_usd"] * float(rng.uniform(*SLA_CREDIT_RANGE)), 2)
            add(inv, credit, tl.incident_end + _days(rng, 7, 31), "sla_credit")

    records.sort(key=lambda r: (r["refund_date"], r["invoice_id"]))
    ids = _ids("RF-")
    for r in records:
        r["refund_id"] = next(ids)
    return records


def _opportunities(
    rng: np.random.Generator,
    fake: Faker,
    tl: Timeline,
    prices: PriceBook,
    reps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reps_by_region = defaultdict(list)
    for rep in reps:
        reps_by_region[rep["region"]].append(rep["rep_id"])
    regions = list(REGION_CUSTOMER_SHARE)
    segments = list(SEGMENTS)
    ids = _ids("OPP-")
    records = []
    for i, month in enumerate(tl.invoice_months()):
        for _ in range(int(rng.poisson(90 + 3 * i))):
            region = _pick(rng, regions, list(REGION_CUSTOMER_SHARE.values()))
            segment = _pick(rng, segments, [s.share for s in SEGMENTS.values()])
            seg = SEGMENTS[segment]
            product = _pick(
                rng, [p for p, _ in seg.core_products], [w for _, w in seg.core_products]
            )
            seats = _log_uniform_int(rng, *seg.seats)
            created = month + _days(rng, 0, 28)
            close = created + _days(rng, 15, 91)
            record: dict[str, Any] = {
                "opportunity_id": next(ids),
                "account_name": fake.company(),
                "region": region,
                "country_code": _pick(rng, COUNTRIES[region])[0],
                "segment": segment,
                "product_id": product,
                "amount_usd": round(seats * prices.price(product, region, created) * 12, -2),
                "created_date": created,
                "close_date": close,
                "lost_reason": None,
                "competitor": None,
                "owner_rep_id": _pick(rng, reps_by_region[region]),
            }
            if created > tl.as_of:
                continue
            if close > tl.as_of:
                record["stage"] = _pick(
                    rng, ["prospecting", "qualification", "proposal", "negotiation"]
                )
            else:
                record.update(_close_outcome(rng, tl, region, close))
            records.append(record)
    return records


def _close_outcome(
    rng: np.random.Generator, tl: Timeline, region: str, close: date
) -> dict[str, Any]:
    in_drop = tl.drop_q_start <= close < tl.current_q_start
    win_rate, reason_weights = 0.28, [0.20, 0.30, 0.30, 0.20]
    competitor_weights = [0.35, 0.35, 0.20, 0.10 if region == "APAC" else 0.02]
    if in_drop and region == "EMEA":
        win_rate, reason_weights = 0.18, [0.55, 0.20, 0.15, 0.10]
    elif in_drop and region == "APAC":
        win_rate, reason_weights = 0.15, [0.15, 0.60, 0.15, 0.10]
        competitor_weights = [0.10, 0.10, 0.10, 0.70]
    if rng.random() < win_rate:
        return {"stage": "closed_won"}
    reason = _pick(rng, LOST_REASONS, reason_weights)
    competitor = _pick(rng, COMPETITORS, competitor_weights) if reason == "competitor" else None
    return {"stage": "closed_lost", "lost_reason": reason, "competitor": competitor}


def _tickets(
    rng: np.random.Generator,
    tl: Timeline,
    customers: dict[str, Customer],
    subs: list[Subscription],
) -> list[dict[str, Any]]:
    subs_by_customer: dict[str, list[Subscription]] = defaultdict(list)
    for s in subs:
        subs_by_customer[s.customer_id].append(s)

    records: list[dict[str, Any]] = []

    def add(
        customer_id: str, created: datetime, category: str, subject: str, *, unhappy: bool
    ) -> None:
        hours = int(rng.integers(2, 72)) if category == "outage" else int(rng.integers(1, 120))
        resolved = created + timedelta(hours=hours)
        is_resolved = resolved.date() <= tl.as_of
        csat = None
        if is_resolved and rng.random() < 0.6:
            weights = [0.35, 0.30, 0.20, 0.10, 0.05] if unhappy else [0.03, 0.07, 0.15, 0.40, 0.35]
            csat = _pick(rng, [1, 2, 3, 4, 5], weights)
        records.append(
            {
                "customer_id": customer_id,
                "created_at": created,
                "category": category,
                "priority": _priority(rng, category),
                "subject": subject,
                "status": "closed" if is_resolved else "open",
                "resolved_at": resolved if is_resolved else None,
                "csat_score": csat,
            }
        )

    def moment(start: date, end: date) -> datetime:
        span = max((end - start).days, 1)
        day = start + _days(rng, 0, span)
        return datetime.combine(day, time(int(rng.integers(0, 24)), int(rng.integers(0, 60))))

    categories = ["how_to", "bug", "billing", "outage"]
    for c in customers.values():
        own = subs_by_customer[c.customer_id]
        rate = SEGMENTS[c.segment].monthly_tickets
        for m in tl.invoice_months():
            if not any(s.is_active_on(m) for s in own):
                continue
            for _ in range(int(rng.poisson(rate))):
                category = _pick(rng, categories, [0.40, 0.30, 0.20, 0.10])
                created = moment(m, min(add_months(m, 1), tl.as_of))
                add(
                    c.customer_id,
                    created,
                    category,
                    _pick(rng, TICKET_SUBJECTS[category]),
                    unhappy=False,
                )

        if c.region == "APAC" and any(s.is_active_on(tl.incident_start) for s in own):
            for _ in range(int(rng.poisson(1.5))):
                created = moment(tl.incident_start, tl.incident_end)
                add(c.customer_id, created, "outage", _pick(rng, INCIDENT_SUBJECTS), unhappy=True)

        if c.region == "EMEA" and any(
            s.product_id == "P02" and s.is_active_on(tl.emea_memo_date) for s in own
        ):
            for _ in range(int(rng.poisson(0.8))):
                created = moment(tl.emea_memo_date, tl.drop_q_start + timedelta(days=45))
                add(c.customer_id, created, "billing", _pick(rng, PRICE_SUBJECTS), unhappy=True)

    records = [r for r in records if r["created_at"].date() <= tl.as_of]
    records.sort(key=lambda r: (r["created_at"], r["customer_id"]))
    ids = _ids("T-")
    for r in records:
        r["ticket_id"] = next(ids)
    return records


def _priority(rng: np.random.Generator, category: str) -> str:
    options = {
        "outage": (["critical", "high"], [0.6, 0.4]),
        "bug": (["high", "medium", "low"], [0.3, 0.5, 0.2]),
        "billing": (["medium", "low"], [0.5, 0.5]),
        "how_to": (["low", "medium"], [0.8, 0.2]),
    }
    levels, weights = options[category]
    return _pick(rng, levels, weights)


def _customer_frame(customers: dict[str, Customer], subs: list[Subscription]) -> pd.DataFrame:
    ends: dict[str, list[date | None]] = defaultdict(list)
    for s in subs:
        ends[s.customer_id].append(s.end_date)
    rows = []
    for c in customers.values():
        customer_ends = ends[c.customer_id]
        churned = all(e is not None for e in customer_ends)
        churn_date = max(e for e in customer_ends if e is not None) if churned else None
        rows.append(
            {
                **asdict(c),
                "status": "churned" if churned else "active",
                "churn_date": churn_date,
            }
        )
    return pd.DataFrame(rows)


def revenue_by_region(ds: Dataset) -> pd.DataFrame:
    """Gross and net revenue per region for the comparison quarter and the drop quarter.

    Net revenue = invoices dated in the quarter minus refunds dated in the quarter.
    """
    tl = ds.timeline
    inv = ds.invoices
    refunds = ds.refunds.merge(inv[["invoice_id", "region"]], on="invoice_id")
    table: dict[str, dict[str, float]] = {region: {} for region in REGIONS}
    for label, start in (("prev", tl.prev_q_start), ("drop", tl.drop_q_start)):
        end = add_months(start, 3)
        in_q = inv[(inv["invoice_date"] >= start) & (inv["invoice_date"] < end)]
        rf_q = refunds[(refunds["refund_date"] >= start) & (refunds["refund_date"] < end)]
        gross = in_q.groupby("region")["amount_usd"].sum()
        refunded = rf_q.groupby("region")["amount_usd"].sum()
        for region in REGIONS:
            g = float(gross.get(region, 0.0))
            table[region][f"gross_{label}"] = g
            table[region][f"net_{label}"] = g - float(refunded.get(region, 0.0))
    frame = pd.DataFrame.from_dict(table, orient="index")
    frame.index.name = "region"
    frame["net_change_pct"] = (frame["net_drop"] / frame["net_prev"] - 1) * 100
    return frame
