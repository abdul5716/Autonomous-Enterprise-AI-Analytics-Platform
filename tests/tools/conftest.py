from datetime import date
from pathlib import Path

import duckdb
import pytest

from analytics_platform.config import Settings

AS_OF = date(2026, 11, 26)

# Two regions, two quarters: Q2 2026 and Q3 2026, plus a budget for Q3.
REVENUE_ROWS = [
    # date_key, customer_key, product_key, region_key, txn_type, amount_usd, seats, refund_reason
    (20260401, 1, 1, 1, "invoice", 1000.0, 10, None),
    (20260501, 2, 1, 2, "invoice", 500.0, 5, None),
    (20260701, 1, 1, 1, "invoice", 900.0, 9, None),
    (20260701, 2, 1, 2, "invoice", 300.0, 3, None),
    (20260715, 2, 1, 2, "refund", -50.0, 0, "sla_credit"),
]


@pytest.fixture
def warehouse_settings(tmp_path: Path) -> Settings:
    path = tmp_path / "wh.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        """
        CREATE TABLE dim_date AS
        SELECT CAST(strftime(d, '%Y%m%d') AS INTEGER) AS date_key, CAST(d AS DATE) AS full_date,
               year(d) AS year, quarter(d) AS quarter, month(d) AS month,
               strftime(d, '%Y-%m') AS year_month
        FROM range(DATE '2025-01-01', DATE '2027-01-01', INTERVAL 1 DAY) t(d)
        """
    )
    con.execute(
        "CREATE TABLE dim_region AS SELECT * FROM (VALUES "
        "(1, 'NA', 'North America'), (2, 'EMEA', 'Europe')) t(region_key, region_code, region_name)"
    )
    con.execute(
        "CREATE TABLE dim_product AS SELECT * FROM (VALUES "
        "(1, 'P02', 'Nimbus CRM Professional', 'Core CRM', 'Professional')) "
        "t(product_key, product_id, product_name, product_line, edition)"
    )
    con.execute(
        "CREATE TABLE dim_customer AS SELECT * FROM (VALUES "
        "(1, 'C1', 'Acme', 'SMB', 'Retail', 'United States'), "
        "(2, 'C2', 'Globex', 'Enterprise', 'Media', 'Germany')) "
        "t(customer_key, customer_id, company_name, segment, industry, country_name)"
    )
    con.execute(
        "CREATE TABLE fact_revenue (date_key INTEGER, customer_key INTEGER, product_key INTEGER, "
        "region_key INTEGER, txn_type VARCHAR, amount_usd DOUBLE, seats INTEGER, "
        "refund_reason VARCHAR)"
    )
    con.executemany("INSERT INTO fact_revenue VALUES (?, ?, ?, ?, ?, ?, ?, ?)", REVENUE_ROWS)
    con.execute(
        "CREATE TABLE fact_budget AS SELECT * FROM (VALUES "
        "(20260701, 1, 'Core CRM', 1100.0), (20260701, 2, 'Core CRM', 600.0)) "
        "t(date_key, region_key, product_line, amount_usd)"
    )
    con.execute(f"CREATE TABLE warehouse_info AS SELECT DATE '{AS_OF}' AS data_as_of")
    con.close()
    return Settings(_env_file=None, duckdb_path=str(path), llm_retry_initial_wait_s=0)
