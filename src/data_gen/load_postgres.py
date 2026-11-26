"""Load the CRM tables into Postgres (schema `crm`). Re-running replaces the data."""

import psycopg

from data_gen.frames import rows
from data_gen.simulate import Dataset

DDL = """
DROP TABLE IF EXISTS crm.support_tickets, crm.opportunities, crm.subscriptions,
    crm.price_history, crm.customers, crm.products, crm.sales_reps, crm.countries CASCADE;

CREATE TABLE crm.countries (
    country_code  char(2) PRIMARY KEY,
    country_name  text NOT NULL,
    region        text NOT NULL
);
COMMENT ON TABLE crm.countries IS
    'Countries and their sales region (NA, EMEA, APAC, LATAM)';

CREATE TABLE crm.sales_reps (
    rep_id      text PRIMARY KEY,
    full_name   text NOT NULL,
    email       text NOT NULL UNIQUE,
    region      text NOT NULL,
    hired_date  date NOT NULL
);

CREATE TABLE crm.products (
    product_id     text PRIMARY KEY,
    product_name   text NOT NULL,
    product_line   text NOT NULL,
    edition        text NOT NULL,
    launched_date  date
);

CREATE TABLE crm.price_history (
    product_id      text NOT NULL REFERENCES crm.products,
    region          text NOT NULL,
    unit_price_usd  numeric(10, 2) NOT NULL,
    effective_from  date NOT NULL,
    effective_to    date,
    PRIMARY KEY (product_id, region, effective_from)
);
COMMENT ON TABLE crm.price_history IS
    'List price per seat per month by product and region. '
    'effective_to is exclusive; NULL means current';

CREATE TABLE crm.customers (
    customer_id   text PRIMARY KEY,
    company_name  text NOT NULL,
    industry      text NOT NULL,
    segment       text NOT NULL,
    country_code  char(2) NOT NULL REFERENCES crm.countries,
    region        text NOT NULL,
    signup_date   date NOT NULL,
    owner_rep_id  text NOT NULL REFERENCES crm.sales_reps,
    status        text NOT NULL,
    churn_date    date
);
COMMENT ON COLUMN crm.customers.segment IS 'SMB, Mid-Market or Enterprise';

CREATE TABLE crm.subscriptions (
    subscription_id  text PRIMARY KEY,
    customer_id      text NOT NULL REFERENCES crm.customers,
    product_id       text NOT NULL REFERENCES crm.products,
    seats            integer NOT NULL,
    discount_pct     numeric(5, 3) NOT NULL,
    start_date       date NOT NULL,
    end_date         date,
    cancel_reason    text,
    competitor       text
);
COMMENT ON COLUMN crm.subscriptions.cancel_reason IS
    'Why the subscription ended: price_increase, switched_vendor, service_reliability, downgrade, '
    'budget_cuts, business_closed, low_usage. downgrade means it was replaced by a smaller one';
COMMENT ON COLUMN crm.subscriptions.competitor IS 'Vendor the customer moved to, when known';

CREATE TABLE crm.opportunities (
    opportunity_id  text PRIMARY KEY,
    account_name    text NOT NULL,
    region          text NOT NULL,
    country_code    char(2) NOT NULL REFERENCES crm.countries,
    segment         text NOT NULL,
    product_id      text NOT NULL REFERENCES crm.products,
    amount_usd      numeric(14, 2) NOT NULL,
    stage           text NOT NULL,
    created_date    date NOT NULL,
    close_date      date NOT NULL,
    lost_reason     text,
    competitor      text,
    owner_rep_id    text NOT NULL REFERENCES crm.sales_reps
);
COMMENT ON COLUMN crm.opportunities.amount_usd IS 'Annual contract value';
COMMENT ON COLUMN crm.opportunities.close_date IS
    'Actual close date, or expected close date if open';

CREATE TABLE crm.support_tickets (
    ticket_id    text PRIMARY KEY,
    customer_id  text NOT NULL REFERENCES crm.customers,
    created_at   timestamp NOT NULL,
    category     text NOT NULL,
    priority     text NOT NULL,
    subject      text NOT NULL,
    status       text NOT NULL,
    resolved_at  timestamp,
    csat_score   smallint
);

CREATE INDEX ON crm.subscriptions (customer_id);
CREATE INDEX ON crm.opportunities (close_date);
CREATE INDEX ON crm.support_tickets (customer_id, created_at);

-- Row-level security for the agent's read-only role. The SQL tool sets app.region_scope per
-- transaction: '*' for all regions or a comma-separated list. Unset means no rows (fail closed).
CREATE OR REPLACE FUNCTION crm.region_visible(r text) RETURNS boolean
LANGUAGE sql STABLE AS $$
    SELECT CASE coalesce(current_setting('app.region_scope', true), '')
        WHEN '' THEN false
        WHEN '*' THEN true
        ELSE r = ANY (string_to_array(current_setting('app.region_scope', true), ','))
    END
$$;

ALTER TABLE crm.customers ENABLE ROW LEVEL SECURITY;
ALTER TABLE crm.sales_reps ENABLE ROW LEVEL SECURITY;
ALTER TABLE crm.opportunities ENABLE ROW LEVEL SECURITY;
ALTER TABLE crm.subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE crm.support_tickets ENABLE ROW LEVEL SECURITY;

CREATE POLICY region_scope ON crm.customers FOR SELECT TO analyst_ro
    USING (crm.region_visible(region));
CREATE POLICY region_scope ON crm.sales_reps FOR SELECT TO analyst_ro
    USING (crm.region_visible(region));
CREATE POLICY region_scope ON crm.opportunities FOR SELECT TO analyst_ro
    USING (crm.region_visible(region));
CREATE POLICY region_scope ON crm.subscriptions FOR SELECT TO analyst_ro
    USING (EXISTS (SELECT 1 FROM crm.customers c WHERE c.customer_id = subscriptions.customer_id));
CREATE POLICY region_scope ON crm.support_tickets FOR SELECT TO analyst_ro
    USING (EXISTS (
        SELECT 1 FROM crm.customers c WHERE c.customer_id = support_tickets.customer_id
    ));
"""

TABLES: dict[str, list[str]] = {
    "countries": ["country_code", "country_name", "region"],
    "sales_reps": ["rep_id", "full_name", "email", "region", "hired_date"],
    "products": ["product_id", "product_name", "product_line", "edition", "launched_date"],
    "price_history": ["product_id", "region", "unit_price_usd", "effective_from", "effective_to"],
    "customers": [
        "customer_id",
        "company_name",
        "industry",
        "segment",
        "country_code",
        "region",
        "signup_date",
        "owner_rep_id",
        "status",
        "churn_date",
    ],
    "subscriptions": [
        "subscription_id",
        "customer_id",
        "product_id",
        "seats",
        "discount_pct",
        "start_date",
        "end_date",
        "cancel_reason",
        "competitor",
    ],
    "opportunities": [
        "opportunity_id",
        "account_name",
        "region",
        "country_code",
        "segment",
        "product_id",
        "amount_usd",
        "stage",
        "created_date",
        "close_date",
        "lost_reason",
        "competitor",
        "owner_rep_id",
    ],
    "support_tickets": [
        "ticket_id",
        "customer_id",
        "created_at",
        "category",
        "priority",
        "subject",
        "status",
        "resolved_at",
        "csat_score",
    ],
}


def load_postgres(ds: Dataset, dsn: str) -> dict[str, int]:
    counts = {}
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        for table, columns in TABLES.items():
            df = getattr(ds, table)
            with cur.copy(f"COPY crm.{table} ({', '.join(columns)}) FROM STDIN") as copy:
                for row in rows(df, columns):
                    copy.write_row(row)
            counts[f"crm.{table}"] = len(df)
        cur.execute("ANALYZE")
    return counts
