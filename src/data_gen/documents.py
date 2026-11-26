"""Generate Nimbus Analytics' internal documents as PDFs, plus a metadata manifest.

Dates, prices and figures come from the simulated dataset, so documents and databases agree.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from data_gen.catalog import (
    EMEA_PROFESSIONAL_INCREASE,
    FICTIONAL_COMPETITOR,
    PRODUCTS,
    REGIONS,
)
from data_gen.simulate import Dataset
from data_gen.timeline import add_months, month_start

ALL_EMPLOYEES = ["all"]


@dataclass
class Section:
    heading: str
    paragraphs: list[str] = field(default_factory=list)
    table: list[list[str]] | None = None


@dataclass
class Document:
    doc_id: str
    title: str
    doc_type: str
    version: str
    effective_date: date
    owner: str
    acl: list[str]
    status: str
    sections: list[Section]
    supersedes: str | None = None

    @property
    def filename(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", self.title.lower()).strip("-")
        return f"{self.doc_id.lower()}-v{self.version}-{slug}.pdf"

    def metadata(self) -> dict[str, object]:
        meta = asdict(self)
        del meta["sections"]
        meta["effective_date"] = self.effective_date.isoformat()
        meta["filename"] = self.filename
        return meta


def fmt_date(d: date) -> str:
    return f"{d.day} {d:%B %Y}"


def usd(amount: float, decimals: int = 2) -> str:
    return f"${amount:,.{decimals}f}"


def usd_m(amount: float) -> str:
    return f"${amount / 1e6:,.2f}M"


def quarter_net_by_region(ds: Dataset, q_start: date) -> dict[str, float]:
    end = add_months(q_start, 3)
    inv = ds.invoices[(ds.invoices.invoice_date >= q_start) & (ds.invoices.invoice_date < end)]
    refunds = ds.refunds.merge(ds.invoices[["invoice_id", "region"]], on="invoice_id")
    refunds = refunds[(refunds.refund_date >= q_start) & (refunds.refund_date < end)]
    gross = inv.groupby("region").amount_usd.sum()
    refunded = refunds.groupby("region").amount_usd.sum()
    return {r: float(gross.get(r, 0.0) - refunded.get(r, 0.0)) for r in REGIONS}


def build_documents(ds: Dataset) -> list[Document]:
    return [
        _refund_policy_v3(ds),
        _refund_policy_v2(ds),
        _sla_policy(ds),
        _discount_policy(ds),
        _emea_pricing_memo(ds),
        _apac_postmortem(ds),
        _orbit_field_report(ds),
        _quarterly_business_review(ds),
        _editions_overview(ds),
        _ai_assistant_guide(ds),
        _analytics_addon_guide(ds),
        _privacy_policy(ds),
        _battlecard(ds),
        _onboarding_guide(ds),
        _travel_policy(ds),
    ]


def _refund_policy_v3(ds: Dataset) -> Document:
    return Document(
        doc_id="POL-FIN-007",
        title="Refund and Cancellation Policy",
        doc_type="policy",
        version="3.0",
        effective_date=add_months(ds.timeline.history_start, 15),
        owner="Finance Operations",
        acl=ALL_EMPLOYEES,
        status="current",
        supersedes="POL-FIN-007 v2.0",
        sections=[
            Section(
                "1. Purpose and scope",
                [
                    "This policy defines how Nimbus Analytics handles subscription cancellations "
                    "and refunds for all Nimbus CRM editions and add-ons purchased directly from "
                    "Nimbus. It replaces version 2.0 of this policy in full.",
                ],
            ),
            Section(
                "2. Definitions",
                [
                    "<b>Enterprise customer</b>: a customer subscribed to Nimbus CRM Enterprise "
                    "edition, or any customer whose annual contract value exceeds $100,000.",
                    "<b>Standard customer</b>: any customer that is not an Enterprise customer, "
                    "typically SMB and Mid-Market accounts.",
                    "<b>Billing period</b>: subscriptions are invoiced monthly on the first day of "
                    "each month.",
                ],
            ),
            Section(
                "3. Standard customers",
                [
                    "Standard customers may cancel at any time. Cancellation takes effect at the "
                    "end of the current billing month.",
                    "A request made within 14 days of the invoice date receives a full refund of "
                    "that invoice. After 14 days, the current month is not refundable.",
                ],
            ),
            Section(
                "4. Enterprise customers",
                [
                    "Enterprise customers may request a <b>pro-rated refund</b> of unused "
                    "subscription fees within <b>30 days</b> of the invoice date. The refund is "
                    "calculated on a daily pro-rata basis for the remaining days of the billing "
                    "period.",
                    "Any single refund above <b>$50,000</b> requires approval from the "
                    "<b>VP of Finance</b> before it is issued.",
                    "Enterprise contracts with a committed term may be terminated early with 60 "
                    "days' written notice. An early termination fee of 25% of the remaining "
                    "committed fees applies, unless the termination is due to a material breach "
                    "of the Service Level Agreement (see POL-OPS-003).",
                ],
            ),
            Section(
                "5. Approval matrix",
                ["Refund approvals are based on the amount of a single refund:"],
                table=[
                    ["Refund amount", "Approver", "Approval level code"],
                    ["Up to $5,000", "Customer Success Manager", "standard"],
                    ["$5,000.01 to $50,000", "Finance Manager", "finance_manager"],
                    ["Above $50,000", "VP of Finance", "vp_finance"],
                ],
            ),
            Section(
                "6. Service credits",
                [
                    "Service credits owed under the SLA and Service Credit Policy (POL-OPS-003) "
                    "are issued as refunds against the affected month's invoice, using reason "
                    "code sla_credit. They do not count against a customer's refund eligibility.",
                ],
            ),
            Section(
                "7. Processing",
                [
                    "Approved refunds are paid to the original payment method within 10 business "
                    "days and recorded in the finance system with one of these reason codes: "
                    "billing_error, goodwill_credit, prorated_cancellation, sla_credit.",
                    "Professional services and training fees are non-refundable.",
                ],
            ),
        ],
    )


def _refund_policy_v2(ds: Dataset) -> Document:
    return Document(
        doc_id="POL-FIN-007",
        title="Refund and Cancellation Policy",
        doc_type="policy",
        version="2.0",
        effective_date=add_months(ds.timeline.history_start, -12),
        owner="Finance Operations",
        acl=ALL_EMPLOYEES,
        status="superseded",
        sections=[
            Section(
                "Status",
                ["SUPERSEDED. This version was replaced by version 3.0 and is kept for audit."],
            ),
            Section(
                "1. Cancellations",
                [
                    "All customers may cancel with 30 days' notice. Cancellation takes effect at "
                    "the end of the following billing month.",
                ],
            ),
            Section(
                "2. Refunds",
                [
                    "Refunds are available only for billing errors, within 14 days of the invoice "
                    "date, for all customer types including enterprise customers.",
                    "Pro-rated refunds are not offered. Any refund above $25,000 requires approval "
                    "from the Chief Financial Officer.",
                ],
            ),
        ],
    )


def _sla_policy(ds: Dataset) -> Document:
    return Document(
        doc_id="POL-OPS-003",
        title="SLA and Service Credit Policy",
        doc_type="policy",
        version="1.4",
        effective_date=add_months(ds.timeline.history_start, 6),
        owner="Customer Operations",
        acl=ALL_EMPLOYEES,
        status="current",
        sections=[
            Section(
                "1. Availability commitment",
                [
                    "Nimbus commits to monthly availability of 99.9% for Professional edition and "
                    "99.95% for Enterprise edition. Starter edition has no availability "
                    "commitment. Availability is measured per hosting region.",
                ],
            ),
            Section(
                "2. Service credits",
                ["If monthly availability falls below the commitment, customers receive:"],
                table=[
                    ["Monthly availability", "Credit (% of monthly fee)"],
                    ["99.0% to below commitment", "10%"],
                    ["95.0% to below 99.0%", "25%"],
                    ["Below 95.0%", "50%"],
                ],
            ),
            Section(
                "3. Claiming credits",
                [
                    "Customers must request credits within 30 days of the end of the affected "
                    "month. For major incidents declared by Nimbus, credits are applied "
                    "automatically to all eligible customers in the affected region.",
                    "Credits are issued as refunds against the affected invoice (reason code "
                    "sla_credit) and are the customer's sole remedy for unavailability.",
                ],
            ),
        ],
    )


def _discount_policy(ds: Dataset) -> Document:
    return Document(
        doc_id="POL-SAL-011",
        title="Discount Approval Policy",
        doc_type="policy",
        version="2.1",
        effective_date=add_months(ds.timeline.history_start, 3),
        owner="Sales Operations",
        acl=["sales", "finance", "leadership"],
        status="current",
        sections=[
            Section(
                "1. Discount bands",
                ["Discounts are a percentage off list price and require approval as follows:"],
                table=[
                    ["Segment", "Discount", "Approver"],
                    ["SMB", "Up to 5%", "Account Executive"],
                    ["Mid-Market", "5% to 15%", "Sales Manager"],
                    ["Enterprise", "Up to 30%", "Regional VP of Sales"],
                    ["Any", "Above 30%", "Chief Financial Officer"],
                ],
            ),
            Section(
                "2. Retention discounts",
                [
                    "Retention discounts offered to prevent churn follow the same bands and must "
                    "be logged against the account with reason code retention.",
                ],
            ),
        ],
    )


def _emea_pricing_memo(ds: Dataset) -> Document:
    tl = ds.timeline
    day_before = tl.drop_q_start - timedelta(days=1)
    old = ds.prices.price("P02", "EMEA", day_before)
    new = ds.prices.price("P02", "EMEA", tl.drop_q_start)
    increase_pct = round(EMEA_PROFESSIONAL_INCREASE * 100)
    rows = [["Edition", "Current price", "New price"]]
    for product in PRODUCTS[:3]:
        before = ds.prices.price(product.product_id, "EMEA", day_before)
        after = ds.prices.price(product.product_id, "EMEA", tl.drop_q_start)
        rows.append([product.name, usd(before), usd(after)])
    return Document(
        doc_id=f"MEMO-PRC-{tl.emea_memo_date.year}-04",
        title="EMEA Pricing Update for Nimbus CRM Professional",
        doc_type="memo",
        version="1.0",
        effective_date=tl.emea_memo_date,
        owner="Pricing Committee",
        acl=["sales", "finance", "leadership"],
        status="current",
        sections=[
            Section(
                "Decision",
                [
                    f"Effective <b>{fmt_date(tl.drop_q_start)}</b>, the EMEA list price of "
                    f"Nimbus CRM Professional increases by {increase_pct}%, from {usd(old)} to "
                    f"{usd(new)} per seat per month. The new price applies to new and existing "
                    "subscriptions from the first invoice on or after the effective date.",
                ],
                table=rows,
            ),
            Section(
                "Rationale",
                [
                    "EMEA Professional pricing has been well below North America for two years. "
                    "The increase funds the EU data residency programme and aligns regional "
                    "margins.",
                ],
            ),
            Section(
                "Risks",
                [
                    "The committee estimates that up to 8% of EMEA Professional customers may "
                    "churn or downgrade. EMEA Sales leadership flagged that the risk is higher "
                    "for price-sensitive SMB and Mid-Market accounts, which make up most of the "
                    "EMEA Professional base.",
                ],
            ),
            Section(
                "Mitigation",
                [
                    "Account Executives may offer a 10% retention discount for 12 months to "
                    "accounts with more than three years of tenure, under the Discount Approval "
                    "Policy (POL-SAL-011).",
                ],
            ),
        ],
    )


def _apac_postmortem(ds: Dataset) -> Document:
    tl = ds.timeline
    customers = ds.customers.set_index("customer_id")
    tickets = ds.support_tickets
    in_window = tickets[
        (tickets.category == "outage")
        & (tickets.created_at.dt.date >= tl.incident_start)
        & (tickets.created_at.dt.date <= tl.incident_end)
        & (tickets.customer_id.map(customers.region) == "APAC")
    ]
    credits = ds.refunds[ds.refunds.reason == "sla_credit"]
    year = tl.incident_start.year
    return Document(
        doc_id=f"INC-{year}-017",
        title="Postmortem: APAC Platform Outage",
        doc_type="postmortem",
        version="1.0",
        effective_date=tl.incident_end + timedelta(days=6),
        owner="Site Reliability Engineering",
        acl=ALL_EMPLOYEES,
        status="current",
        sections=[
            Section(
                "Summary",
                [
                    f"On {fmt_date(tl.incident_start)} at 02:10 SGT, the APAC hosting region "
                    "(Singapore) suffered a full outage lasting 9 hours, followed by degraded "
                    f"performance until {fmt_date(tl.incident_end)}. Monthly availability for "
                    "APAC fell to 98.7%, below the 99.9% Professional and 99.95% Enterprise "
                    "commitments.",
                ],
            ),
            Section(
                "Impact",
                [
                    f"All APAC customers were affected. Support received {len(in_window)} outage "
                    "tickets during the incident window, most at critical priority.",
                    f"Under POL-OPS-003, Nimbus automatically issued SLA credits of 10% to 25% "
                    f"of the monthly fee to {credits.customer_id.nunique()} Mid-Market and "
                    f"Enterprise customers, totalling {usd(float(credits.amount_usd.sum()), 0)}.",
                ],
            ),
            Section(
                "Root cause",
                [
                    "A storage migration on the primary database cluster failed midway. "
                    "Automatic failover to the standby cluster did not trigger because a "
                    "health-check threshold had been misconfigured during the migration.",
                ],
            ),
            Section(
                "Timeline (SGT)",
                table=[
                    ["Time", "Event"],
                    [f"{fmt_date(tl.incident_start)} 02:10", "Database migration step fails"],
                    [f"{fmt_date(tl.incident_start)} 02:25", "Alerts fire, incident declared"],
                    [f"{fmt_date(tl.incident_start)} 11:10", "Manual failover, service restored"],
                    [f"{fmt_date(tl.incident_end)} 18:00", "Performance back to normal"],
                ],
            ),
            Section(
                "Follow-up actions",
                [
                    "Automate failover testing before every migration; add a second standby in "
                    "a separate availability zone; customer success to contact every affected "
                    "Enterprise account within two weeks.",
                ],
            ),
        ],
    )


def _orbit_field_report(ds: Dataset) -> Document:
    tl = ds.timeline
    return Document(
        doc_id=f"MEMO-SAL-{tl.orbit_promo_start.year}-09",
        title=f"APAC Field Report: {FICTIONAL_COMPETITOR} Promotional Campaign",
        doc_type="memo",
        version="1.0",
        effective_date=tl.incident_end + timedelta(days=10),
        owner="APAC Sales",
        acl=["sales", "leadership"],
        status="current",
        sections=[
            Section(
                "What happened",
                [
                    f"From {fmt_date(tl.orbit_promo_start)}, {FICTIONAL_COMPETITOR}, a "
                    "Singapore-based CRM vendor, launched a 'Switch and Save' campaign in APAC: "
                    "50% off the first year plus free data migration, aimed at Nimbus "
                    "Professional and Enterprise customers.",
                ],
            ),
            Section(
                "Field observations",
                [
                    "Account teams report that several APAC Mid-Market and Enterprise accounts "
                    f"have requested {FICTIONAL_COMPETITOR} demos. Since the "
                    f"{fmt_date(tl.incident_start)} outage, {FICTIONAL_COMPETITOR} sales reps "
                    "have been citing Nimbus reliability in their pitches.",
                ],
            ),
            Section(
                "Recommendations",
                [
                    "Approve targeted retention discounts for at-risk APAC accounts, prioritise "
                    "executive outreach to Enterprise customers affected by the outage, and "
                    "accelerate the APAC second-zone deployment.",
                ],
            ),
        ],
    )


def _quarterly_business_review(ds: Dataset) -> Document:
    tl = ds.timeline
    before_prev = add_months(tl.prev_q_start, -3)
    current = quarter_net_by_region(ds, tl.prev_q_start)
    previous = quarter_net_by_region(ds, before_prev)
    rows = [["Region", tl.prev_q_label, "Quarter-over-quarter"]]
    for region in REGIONS:
        change = (current[region] / previous[region] - 1) * 100
        rows.append([region, usd_m(current[region]), f"{change:+.1f}%"])
    total, total_before = sum(current.values()), sum(previous.values())
    rows.append(["Total", usd_m(total), f"{(total / total_before - 1) * 100:+.1f}%"])
    budget = ds.budgets[
        (ds.budgets.budget_month >= tl.drop_q_start)
        & (ds.budgets.budget_month < tl.current_q_start)
    ].amount_usd.sum()
    return Document(
        doc_id=f"QBR-{tl.prev_q_label.replace(' ', '-')}",
        title=f"Quarterly Business Review {tl.prev_q_label}",
        doc_type="report",
        version="1.0",
        effective_date=tl.drop_q_start + timedelta(days=20),
        owner="FP&A",
        acl=["leadership", "finance"],
        status="current",
        sections=[
            Section(
                "Revenue",
                [f"Net revenue (invoices less refunds) by region for {tl.prev_q_label}:"],
                table=rows,
            ),
            Section(
                "Outlook and risks",
                [
                    f"The {tl.drop_q_label} revenue budget is {usd_m(float(budget))}.",
                    f"Key risks: the EMEA Professional price increase effective "
                    f"{fmt_date(tl.drop_q_start)} may raise churn in price-sensitive segments; "
                    f"APAC faces new competitive pressure from {FICTIONAL_COMPETITOR}.",
                ],
            ),
        ],
    )


def _editions_overview(ds: Dataset) -> Document:
    as_of = ds.timeline.as_of
    rows = [["Product", *REGIONS]]
    for product in PRODUCTS:
        rows.append(
            [product.name, *(usd(ds.prices.price(product.product_id, r, as_of)) for r in REGIONS)]
        )
    return Document(
        doc_id="PRD-CRM-001",
        title="Nimbus CRM Editions Overview",
        doc_type="product_doc",
        version="5.2",
        effective_date=month_start(as_of),
        owner="Product Marketing",
        acl=ALL_EMPLOYEES,
        status="current",
        sections=[
            Section(
                "Editions",
                [
                    "<b>Starter</b>: contact and deal management, email integration, standard "
                    "reports. For teams of up to 50 users. No availability commitment.",
                    "<b>Professional</b>: everything in Starter plus workflow automation, custom "
                    "reports, API access and a 99.9% availability SLA.",
                    "<b>Enterprise</b>: everything in Professional plus SSO/SAML, sandbox "
                    "environments, advanced permissions, a dedicated customer success manager, "
                    "EU/US/APAC data residency and a 99.95% availability SLA.",
                ],
            ),
            Section(
                "Current list prices",
                ["USD per seat per month, by sales region:"],
                table=rows,
            ),
        ],
    )


def _ai_assistant_guide(ds: Dataset) -> Document:
    return Document(
        doc_id="PRD-AI-002",
        title="Nimbus AI Assistant Add-on Guide",
        doc_type="product_doc",
        version="1.3",
        effective_date=ds.timeline.ai_assistant_launch,
        owner="Product Management",
        acl=ALL_EMPLOYEES,
        status="current",
        sections=[
            Section(
                "Overview",
                [
                    f"Nimbus AI Assistant launched on {fmt_date(ds.timeline.ai_assistant_launch)}. "
                    "It drafts customer emails, summarises deal history, and flags at-risk deals "
                    "in the forecast. Available for Professional and Enterprise editions.",
                ],
            ),
            Section(
                "Data handling",
                [
                    "Customer data is processed in the customer's hosting region and is never "
                    "used to train AI models. Administrators can disable the assistant per team.",
                ],
            ),
        ],
    )


def _analytics_addon_guide(ds: Dataset) -> Document:
    return Document(
        doc_id="PRD-ANL-003",
        title="Nimbus Analytics Add-on Guide",
        doc_type="product_doc",
        version="2.0",
        effective_date=add_months(ds.timeline.history_start, 2),
        owner="Product Management",
        acl=ALL_EMPLOYEES,
        status="current",
        sections=[
            Section(
                "Overview",
                [
                    "The Analytics add-on provides custom dashboards, cohort and funnel analysis, "
                    "and a scheduled export of CRM data to the customer's own data warehouse.",
                ],
            ),
        ],
    )


def _privacy_policy(ds: Dataset) -> Document:
    return Document(
        doc_id="POL-SEC-002",
        title="Data Retention and Privacy Policy",
        doc_type="policy",
        version="3.1",
        effective_date=add_months(ds.timeline.history_start, 9),
        owner="Security and Compliance",
        acl=ALL_EMPLOYEES,
        status="current",
        sections=[
            Section(
                "Retention",
                [
                    "Customer data is kept for the life of the subscription and deleted 90 days "
                    "after termination. Backups are kept for 35 days.",
                ],
            ),
            Section(
                "Residency and compliance",
                [
                    "Data is hosted in Frankfurt (EU), Virginia (US) or Singapore (APAC). Nimbus "
                    "signs a Data Processing Agreement with every customer and complies with GDPR.",
                ],
            ),
        ],
    )


def _battlecard(ds: Dataset) -> Document:
    return Document(
        doc_id="SAL-BC-005",
        title="Competitive Battlecard: Salesforce, HubSpot and Zoho",
        doc_type="sales_enablement",
        version="4.0",
        effective_date=add_months(ds.timeline.drop_q_start, -2),
        owner="Product Marketing",
        acl=["sales", "leadership"],
        status="current",
        sections=[
            Section(
                "How to use this battlecard",
                [
                    "Internal positioning based on Nimbus win/loss interviews. Always check "
                    "competitors' current public pricing and announcements before quoting them.",
                ],
            ),
            Section(
                "Salesforce",
                [
                    "Prospects cite its broad ecosystem and enterprise depth. Nimbus wins on "
                    "total cost of ownership and faster implementation.",
                ],
            ),
            Section(
                "HubSpot",
                [
                    "Prospects cite marketing automation and ease of use. Nimbus wins on sales "
                    "workflow depth and predictable per-seat pricing.",
                ],
            ),
            Section(
                "Zoho",
                [
                    "Prospects cite low price and a broad application suite. Nimbus wins on "
                    "reporting, support quality and enterprise security.",
                ],
            ),
            Section(
                FICTIONAL_COMPETITOR,
                [
                    "Regional APAC vendor competing mainly on price. Nimbus wins on product depth "
                    "and global data residency.",
                ],
            ),
        ],
    )


def _onboarding_guide(ds: Dataset) -> Document:
    return Document(
        doc_id="GUIDE-CS-001",
        title="Enterprise Customer Onboarding Guide",
        doc_type="guide",
        version="2.2",
        effective_date=add_months(ds.timeline.history_start, 4),
        owner="Customer Success",
        acl=ALL_EMPLOYEES,
        status="current",
        sections=[
            Section(
                "Onboarding phases",
                [
                    "Week 1: kickoff and success plan. Weeks 2 to 4: SSO set-up, data migration "
                    "and permissions. Weeks 5 to 6: user training and go-live. Week 8: first "
                    "business review with the customer success manager.",
                ],
            ),
            Section(
                "Cancellations during onboarding",
                [
                    "Requests to cancel or for refunds during onboarding follow the Refund and "
                    "Cancellation Policy (POL-FIN-007).",
                ],
            ),
        ],
    )


def _travel_policy(ds: Dataset) -> Document:
    return Document(
        doc_id="HR-POL-004",
        title="Travel and Expense Policy",
        doc_type="policy",
        version="1.1",
        effective_date=add_months(ds.timeline.history_start, 1),
        owner="People Operations",
        acl=ALL_EMPLOYEES,
        status="current",
        sections=[
            Section(
                "Reimbursement",
                [
                    "Employees are refunded for approved business travel within 30 days of "
                    "submitting receipts. Economy class is required for flights under 6 hours. "
                    "Expenses above $2,000 need director approval.",
                ],
            ),
        ],
    )


def render_pdf(doc: Document, path: Path) -> None:
    styles = getSampleStyleSheet()
    pdf = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        title=doc.title,
        author=f"Nimbus Analytics - {doc.owner}",
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        invariant=True,
    )
    meta = [
        ["Document ID", doc.doc_id, "Version", doc.version],
        ["Effective", fmt_date(doc.effective_date), "Status", doc.status.capitalize()],
        ["Owner", doc.owner, "Audience", ", ".join(doc.acl)],
    ]
    story = [
        Paragraph("Nimbus Analytics", styles["Normal"]),
        Paragraph(escape(doc.title), styles["Title"]),
        _table(meta, header=False),
        Spacer(1, 0.6 * cm),
    ]
    for section in doc.sections:
        story.append(Paragraph(escape(section.heading), styles["Heading2"]))
        story.extend(Paragraph(p, styles["BodyText"]) for p in section.paragraphs)
        if section.table:
            story += [Spacer(1, 0.2 * cm), _table(section.table, header=True)]
        story.append(Spacer(1, 0.3 * cm))
    pdf.build(story)


def _table(data: list[list[str]], *, header: bool) -> Table:
    table = Table(data, hAlign="LEFT")
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    table.setStyle(TableStyle(style))
    return table


def write_documents(ds: Dataset, out_dir: Path) -> list[Document]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.pdf"):
        old.unlink()
    docs = build_documents(ds)
    for doc in docs:
        render_pdf(doc, out_dir / doc.filename)
    manifest = [doc.metadata() for doc in docs]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return docs
