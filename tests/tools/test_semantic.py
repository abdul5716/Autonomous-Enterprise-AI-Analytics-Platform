from datetime import date

import pytest

from analytics_platform.security import DEMO_USERS
from analytics_platform.tools.sql.periods import TimeRange, resolve
from analytics_platform.tools.sql.semantic import MetricQuery, SemanticCompiler, SemanticModel


@pytest.fixture(scope="module")
def compiler() -> SemanticCompiler:
    return SemanticCompiler(SemanticModel.load("semantic_layer/metrics.yaml"))


def query(**kwargs: object) -> MetricQuery:
    return MetricQuery.model_validate(kwargs)


def test_catalog_lists_metrics_dimensions_and_budget_limits(compiler: SemanticCompiler) -> None:
    text = compiler.model.catalog_text()

    assert "net_revenue" in text and "segment" in text
    assert "['budget'] only support dimensions" in text


@pytest.mark.parametrize(
    ("q", "problem"),
    [
        (query(metrics=["profit"]), "Unknown metric 'profit'"),
        (query(metrics=["net_revenue"], dimensions=["planet"]), "Unknown dimension 'planet'"),
        (query(metrics=["budget"], dimensions=["segment"]), "'segment' is not available"),
        (
            query(
                metrics=["net_revenue"],
                filters=[{"dimension": "region", "values": ["Europe"]}],
            ),
            "Allowed: NA, EMEA, APAC, LATAM",
        ),
        (
            query(metrics=["net_revenue"], order_by=[{"field": "region"}]),
            "Cannot order by 'region'",
        ),
    ],
)
def test_validation_explains_problems(
    compiler: SemanticCompiler, q: MetricQuery, problem: str
) -> None:
    assert any(problem in e for e in compiler.validate(q))


def test_filter_values_are_case_insensitive_and_bound_as_params(
    compiler: SemanticCompiler,
) -> None:
    q = query(metrics=["net_revenue"], filters=[{"dimension": "region", "values": ["emea"]}])

    compiled = compiler.compile(q, period=None, user=DEMO_USERS["finance.admin"], limit=10)

    assert compiler.validate(q) == []
    assert "r.region_code IN (?)" in compiled.sql
    assert compiled.params == ["EMEA"]


def test_region_restricted_user_gets_forced_filter(compiler: SemanticCompiler) -> None:
    q = query(metrics=["net_revenue"], dimensions=["segment"])

    compiled = compiler.compile(q, period=None, user=DEMO_USERS["emea.sales"], limit=10)

    assert "JOIN dim_region r" in compiled.sql
    assert compiled.params == ["EMEA"]


def test_time_range_and_default_ordering(compiler: SemanticCompiler) -> None:
    q = query(metrics=["net_revenue"], dimensions=["month"])
    period = resolve(TimeRange(period="last_quarter"), date(2026, 11, 26))

    compiled = compiler.compile(q, period=period, user=DEMO_USERS["finance.admin"], limit=10)

    assert "d.full_date >= ? AND d.full_date < ?" in compiled.sql
    assert compiled.params == [date(2026, 7, 1), date(2026, 10, 1)]
    assert "ORDER BY month ASC" in compiled.sql


def test_revenue_and_budget_are_joined_on_dimensions(compiler: SemanticCompiler) -> None:
    q = query(metrics=["net_revenue", "budget"], dimensions=["region"])

    compiled = compiler.compile(q, period=None, user=DEMO_USERS["finance.admin"], limit=10)

    assert "FULL OUTER JOIN m1 ON m0.region IS NOT DISTINCT FROM m1.region" in compiled.sql
    assert "FROM fact_budget b" in compiled.sql
