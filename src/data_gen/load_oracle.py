"""Load the finance tables into Oracle (schema of the app user). Re-running replaces the data."""

import oracledb

from data_gen.frames import rows
from data_gen.simulate import Dataset

DDL = [
    "DROP TABLE IF EXISTS refunds PURGE",
    "DROP TABLE IF EXISTS invoices PURGE",
    "DROP TABLE IF EXISTS budgets PURGE",
    """
    CREATE TABLE invoices (
        invoice_id       VARCHAR2(20) PRIMARY KEY,
        subscription_id  VARCHAR2(20) NOT NULL,
        customer_id      VARCHAR2(20) NOT NULL,
        product_id       VARCHAR2(10) NOT NULL,
        region           VARCHAR2(10) NOT NULL,
        invoice_date     DATE NOT NULL,
        seats            NUMBER(8) NOT NULL,
        unit_price_usd   NUMBER(10, 2) NOT NULL,
        discount_pct     NUMBER(5, 3) NOT NULL,
        amount_usd       NUMBER(14, 2) NOT NULL,
        status           VARCHAR2(10) NOT NULL
    )
    """,
    """
    CREATE TABLE refunds (
        refund_id       VARCHAR2(20) PRIMARY KEY,
        invoice_id      VARCHAR2(20) NOT NULL REFERENCES invoices,
        customer_id     VARCHAR2(20) NOT NULL,
        refund_date     DATE NOT NULL,
        amount_usd      NUMBER(14, 2) NOT NULL,
        reason          VARCHAR2(30) NOT NULL,
        approval_level  VARCHAR2(20) NOT NULL
    )
    """,
    """
    CREATE TABLE budgets (
        budget_month  DATE NOT NULL,
        region        VARCHAR2(10) NOT NULL,
        product_line  VARCHAR2(30) NOT NULL,
        amount_usd    NUMBER(14, 2) NOT NULL,
        PRIMARY KEY (budget_month, region, product_line)
    )
    """,
    "CREATE INDEX invoices_date_idx ON invoices (invoice_date, region)",
    "CREATE INDEX refunds_date_idx ON refunds (refund_date)",
    "COMMENT ON TABLE invoices IS 'Monthly subscription invoices, one per subscription per month'",
    "COMMENT ON TABLE refunds IS 'Refunds and SLA credits issued against invoices'",
    "COMMENT ON TABLE budgets IS 'Monthly revenue budget by region and product line'",
]

TABLES: dict[str, list[str]] = {
    "invoices": [
        "invoice_id",
        "subscription_id",
        "customer_id",
        "product_id",
        "region",
        "invoice_date",
        "seats",
        "unit_price_usd",
        "discount_pct",
        "amount_usd",
        "status",
    ],
    "refunds": [
        "refund_id",
        "invoice_id",
        "customer_id",
        "refund_date",
        "amount_usd",
        "reason",
        "approval_level",
    ],
    "budgets": ["budget_month", "region", "product_line", "amount_usd"],
}

BATCH_SIZE = 5_000


def create_read_only_user(
    *, admin_user: str, admin_password: str, dsn: str, owner: str, ro_user: str, ro_password: str
) -> None:
    """Create the agent's read-only Oracle user with SELECT on the finance schema only."""
    with oracledb.connect(user=admin_user, password=admin_password, dsn=dsn) as conn:
        cur = conn.cursor()
        cur.execute(f'CREATE USER IF NOT EXISTS {ro_user} IDENTIFIED BY "{ro_password}"')
        cur.execute(f"GRANT CREATE SESSION TO {ro_user}")
        cur.execute(f"GRANT SELECT ANY TABLE ON SCHEMA {owner} TO {ro_user}")


def load_oracle(ds: Dataset, *, user: str, password: str, dsn: str) -> dict[str, int]:
    counts = {}
    with oracledb.connect(user=user, password=password, dsn=dsn) as conn:
        cur = conn.cursor()
        for statement in DDL:
            cur.execute(statement)
        for table, columns in TABLES.items():
            df = getattr(ds, table)
            placeholders = ", ".join(f":{i + 1}" for i in range(len(columns)))
            sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            batch = []
            for row in rows(df, columns):
                batch.append(row)
                if len(batch) == BATCH_SIZE:
                    cur.executemany(sql, batch)
                    batch.clear()
            if batch:
                cur.executemany(sql, batch)
            counts[f"oracle.{table}"] = len(df)
        conn.commit()
    return counts
