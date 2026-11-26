"""ETL: extract from the Postgres CRM and Oracle finance systems into a DuckDB star schema."""

import os
from datetime import date
from pathlib import Path

import duckdb
import oracledb
import pandas as pd
import psycopg

from data_gen.catalog import REGIONS


def _read_postgres(dsn: str, sql: str) -> pd.DataFrame:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql)
        columns = [d.name for d in cur.description or []]
        return pd.DataFrame(cur.fetchall(), columns=columns)


def _read_oracle(user: str, password: str, dsn: str, sql: str) -> pd.DataFrame:
    with oracledb.connect(user=user, password=password, dsn=dsn) as conn:
        cur = conn.cursor()
        cur.execute(sql)
        columns = [d[0].lower() for d in cur.description or []]
        return pd.DataFrame(cur.fetchall(), columns=columns)


def _dim_date(start: date, end: date) -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D")
    return pd.DataFrame(
        {
            "date_key": days.strftime("%Y%m%d").astype(int),
            "full_date": days.date,
            "year": days.year,
            "quarter": days.quarter,
            "month": days.month,
            "month_name": days.strftime("%B"),
            "quarter_label": [f"Q{q} {y}" for q, y in zip(days.quarter, days.year, strict=True)],
            "year_month": days.strftime("%Y-%m"),
            "is_month_start": days.is_month_start,
        }
    )


def _date_key(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime("%Y%m%d").astype(int)


def build_warehouse(
    *,
    postgres_dsn: str,
    oracle_user: str,
    oracle_password: str,
    oracle_dsn: str,
    duckdb_path: str,
    as_of: date,
) -> dict[str, int]:
    customers = _read_postgres(
        postgres_dsn,
        "SELECT c.customer_id, c.company_name, c.segment, c.industry, c.country_code, "
        "k.country_name, c.region, c.signup_date, c.status, c.churn_date "
        "FROM crm.customers c JOIN crm.countries k USING (country_code) ORDER BY customer_id",
    )
    products = _read_postgres(
        postgres_dsn,
        "SELECT product_id, product_name, product_line, edition FROM crm.products "
        "ORDER BY product_id",
    )
    invoices = _read_oracle(
        oracle_user,
        oracle_password,
        oracle_dsn,
        "SELECT invoice_id, customer_id, product_id, region, invoice_date, seats, amount_usd "
        "FROM invoices",
    )
    refunds = _read_oracle(
        oracle_user,
        oracle_password,
        oracle_dsn,
        "SELECT r.refund_id, r.refund_date, r.amount_usd, r.reason, "
        "i.customer_id, i.product_id, i.region "
        "FROM refunds r JOIN invoices i ON i.invoice_id = r.invoice_id",
    )
    budgets = _read_oracle(
        oracle_user,
        oracle_password,
        oracle_dsn,
        "SELECT budget_month, region, product_line, amount_usd FROM budgets",
    )

    dim_region = pd.DataFrame(
        {
            "region_key": range(1, len(REGIONS) + 1),
            "region_code": list(REGIONS),
            "region_name": list(REGIONS.values()),
        }
    )
    dim_product = products.reset_index(drop=True)
    dim_product.insert(0, "product_key", range(1, len(dim_product) + 1))
    dim_customer = customers.reset_index(drop=True)
    dim_customer.insert(0, "customer_key", range(1, len(dim_customer) + 1))

    region_keys = dict(zip(dim_region.region_code, dim_region.region_key, strict=True))
    product_keys = dict(zip(dim_product.product_id, dim_product.product_key, strict=True))
    customer_keys = dict(zip(dim_customer.customer_id, dim_customer.customer_key, strict=True))

    invoice_rows = pd.DataFrame(
        {
            "source_id": invoices.invoice_id,
            "txn_type": "invoice",
            "date_key": _date_key(invoices.invoice_date),
            "customer_key": invoices.customer_id.map(customer_keys),
            "product_key": invoices.product_id.map(product_keys),
            "region_key": invoices.region.map(region_keys),
            "amount_usd": invoices.amount_usd.astype(float),
            "seats": invoices.seats.astype(int),
            "refund_reason": None,
        }
    )
    refund_rows = pd.DataFrame(
        {
            "source_id": refunds.refund_id,
            "txn_type": "refund",
            "date_key": _date_key(refunds.refund_date),
            "customer_key": refunds.customer_id.map(customer_keys),
            "product_key": refunds.product_id.map(product_keys),
            "region_key": refunds.region.map(region_keys),
            "amount_usd": -refunds.amount_usd.astype(float),
            "seats": 0,
            "refund_reason": refunds.reason,
        }
    )
    fact_revenue = pd.concat([invoice_rows, refund_rows], ignore_index=True)
    fact_budget = pd.DataFrame(
        {
            "date_key": _date_key(budgets.budget_month),
            "region_key": budgets.region.map(region_keys),
            "product_line": budgets.product_line,
            "amount_usd": budgets.amount_usd.astype(float),
        }
    )
    first_day = pd.to_datetime(customers.signup_date).min().date()
    dim_date = _dim_date(min(first_day, date(as_of.year - 4, 1, 1)), date(as_of.year + 1, 12, 31))

    tables = {
        "dim_date": dim_date,
        "dim_region": dim_region,
        "dim_product": dim_product,
        "dim_customer": dim_customer,
        "fact_revenue": fact_revenue,
        "fact_budget": fact_budget,
    }
    _write_duckdb(Path(duckdb_path), tables, as_of)
    return {name: len(df) for name, df in tables.items()}


def _write_duckdb(path: Path, tables: dict[str, pd.DataFrame], as_of: date) -> None:
    """Build into a temp file, then swap it in, so readers never see a half-built warehouse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    try:
        for name, df in tables.items():
            con.register("staging", df)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM staging")
            con.unregister("staging")
        con.execute(
            "CREATE TABLE warehouse_info AS SELECT ?::DATE AS data_as_of, now() AS built_at",
            [as_of],
        )
        con.execute(
            "COMMENT ON TABLE fact_revenue IS 'Revenue ledger: invoices are positive, refunds and "
            "SLA credits negative. Net revenue = SUM(amount_usd). Grain: one row per invoice or "
            "refund.'"
        )
        con.execute(
            "COMMENT ON TABLE fact_budget IS 'Monthly revenue budget by region and product line'"
        )
    finally:
        con.close()
    os.replace(tmp, path)
