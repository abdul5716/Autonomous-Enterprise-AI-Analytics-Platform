"""Generate all Nimbus Analytics data: `python -m data_gen [--as-of YYYY-MM-DD] [--seed N]`."""

import argparse
import json
import sys
import time
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb

from analytics_platform.config import Settings, get_settings
from data_gen.documents import Document, write_documents
from data_gen.load_oracle import create_read_only_user, load_oracle
from data_gen.load_postgres import load_postgres
from data_gen.simulate import Dataset, revenue_by_region, simulate
from data_gen.warehouse import build_warehouse
from data_gen.web_pages import write_web_fixtures


def _step[R](name: str, fn: Callable[[], R]) -> R:
    print(f"-> {name} ...", flush=True)
    started = time.perf_counter()
    result = fn()
    print(f"   done in {time.perf_counter() - started:.1f}s", flush=True)
    if isinstance(result, dict):
        for key, value in result.items():
            print(f"   {key:<24} {value:>8,}")
    return result


def build_answer_key(ds: Dataset, docs: list[Document], seed: int) -> dict[str, Any]:
    tl = ds.timeline
    by_region = revenue_by_region(ds)
    total_prev, total_drop = by_region.net_prev.sum(), by_region.net_drop.sum()
    doc_ids = {d.doc_type + ":" + d.title: f"{d.doc_id} v{d.version}" for d in docs}

    def doc(title_part: str) -> str:
        return next(v for k, v in doc_ids.items() if title_part in k)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "as_of": tl.as_of.isoformat(),
        "drop_quarter": tl.drop_q_label,
        "comparison_quarter": tl.prev_q_label,
        "current_quarter_is_partial": True,
        "net_revenue": {
            region: {
                "comparison": round(row.net_prev, 2),
                "drop": round(row.net_drop, 2),
                "change_pct": round(row.net_change_pct, 2),
            }
            for region, row in by_region.iterrows()
        },
        "total_net_change_pct": round((total_drop / total_prev - 1) * 100, 2),
        "drivers": {
            "EMEA": {
                "cause": "Professional edition price increase, churn and downgrades",
                "price_increase_effective": tl.drop_q_start.isoformat(),
                "churned_customers": ds.drop_summary.churned["EMEA"],
                "downgraded_customers": ds.drop_summary.downgraded["EMEA"],
                "evidence": [doc("EMEA Pricing"), doc("Quarterly Business Review")],
            },
            "APAC": {
                "cause": "Platform outage plus a competitor promotion; SLA credits reduce net",
                "incident_start": tl.incident_start.isoformat(),
                "churned_customers": ds.drop_summary.churned["APAC"],
                "downgraded_customers": ds.drop_summary.downgraded["APAC"],
                "evidence": [doc("Postmortem"), doc("Field Report")],
            },
        },
        "enterprise_refund_policy": {
            "document": doc("Refund and Cancellation"),
            "refund_window_days": 30,
            "pro_rated": True,
            "vp_finance_approval_above_usd": 50_000,
        },
    }


def verify_warehouse(duckdb_path: str, answer_key: dict[str, Any]) -> dict[str, int]:
    """Net revenue per region in the warehouse must match the simulation to the cent."""
    sql = """
        SELECT r.region_code, d.quarter_label, SUM(f.amount_usd) AS net
        FROM fact_revenue f
        JOIN dim_date d USING (date_key)
        JOIN dim_region r USING (region_key)
        WHERE d.quarter_label IN (?, ?)
        GROUP BY ALL
    """
    with duckdb.connect(duckdb_path, read_only=True) as con:
        result = con.execute(
            sql, [answer_key["comparison_quarter"], answer_key["drop_quarter"]]
        ).fetchall()
    actual = {(region, label): net for region, label, net in result}
    mismatches = 0
    print(f"   {'region':<6} {'comparison':>14} {'drop quarter':>14} {'change':>8}")
    for region, expected in answer_key["net_revenue"].items():
        prev = actual[(region, answer_key["comparison_quarter"])]
        drop = actual[(region, answer_key["drop_quarter"])]
        ok = abs(prev - expected["comparison"]) < 0.01 and abs(drop - expected["drop"]) < 0.01
        mismatches += not ok
        flag = "" if ok else "  MISMATCH"
        print(
            f"   {region:<6} {prev:>14,.0f} {drop:>14,.0f} {(drop / prev - 1) * 100:>7.1f}%{flag}"
        )
    print(f"   total change {answer_key['total_net_change_pct']:+.2f}%")
    if mismatches:
        raise SystemExit(f"Warehouse verification failed for {mismatches} regions")
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--customers", type=int, default=2000)
    args = parser.parse_args()
    s: Settings = get_settings()
    oracle_password = s.oracle_password.get_secret_value()

    ds = _step(
        f"Simulating {args.customers} customers as of {args.as_of}",
        lambda: simulate(args.as_of, seed=args.seed, n_customers=args.customers),
    )
    _step("Loading Postgres CRM", lambda: load_postgres(ds, s.postgres_dsn()))
    _step(
        "Loading Oracle finance",
        lambda: load_oracle(ds, user=s.oracle_user, password=oracle_password, dsn=s.oracle_dsn),
    )
    _step(
        "Creating Oracle read-only user",
        lambda: create_read_only_user(
            admin_user=s.oracle_admin_user,
            admin_password=s.oracle_admin_password.get_secret_value(),
            dsn=s.oracle_dsn,
            owner=s.oracle_user,
            ro_user=s.oracle_ro_user,
            ro_password=s.oracle_ro_password.get_secret_value(),
        ),
    )
    _step(
        "Building DuckDB warehouse from Postgres and Oracle",
        lambda: build_warehouse(
            postgres_dsn=s.postgres_dsn(),
            oracle_user=s.oracle_user,
            oracle_password=oracle_password,
            oracle_dsn=s.oracle_dsn,
            duckdb_path=s.duckdb_path,
            as_of=args.as_of,
        ),
    )
    docs = _step("Writing documents", lambda: write_documents(ds, Path(s.documents_dir)))
    print(f"   {len(docs)} PDFs in {s.documents_dir}")
    pages = _step(
        "Writing offline web fixtures",
        lambda: write_web_fixtures(ds.timeline, Path(s.web_fixtures_dir)),
    )
    print(f"   {len(pages)} pages in {s.web_fixtures_dir}")

    answer_key = build_answer_key(ds, docs, args.seed)
    key_path = Path(s.answer_key_path)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(answer_key, indent=2) + "\n")

    _step(
        "Verifying warehouse against simulation",
        lambda: verify_warehouse(s.duckdb_path, answer_key),
    )
    print(f"Answer key for evals: {key_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
