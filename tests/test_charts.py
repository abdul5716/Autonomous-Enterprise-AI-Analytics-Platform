import pandas as pd
import pytest

from analytics_platform.tools.charts import (
    ChartSpec,
    render_png,
    render_svg,
    suggest_chart,
    to_vega_lite,
    validate_chart,
)

QOQ = pd.DataFrame(
    {
        "region": ["NA", "EMEA"],
        "net_revenue": [21.9e6, 10.7e6],
        "net_revenue_previous": [22.8e6, 15.0e6],
        "net_revenue_change_pct": [-3.9, -28.7],
    }
)
MONTHLY = pd.DataFrame(
    {
        "month": ["2026-07", "2026-08", "2026-07", "2026-08"],
        "region": ["EMEA", "EMEA", "NA", "NA"],
        "net_revenue": [3.8e6, 3.5e6, 7.4e6, 7.3e6],
    }
)


def test_suggests_grouped_bars_for_period_comparisons() -> None:
    spec = suggest_chart(QOQ, "QoQ")

    assert spec is not None
    assert (spec.type, spec.x, spec.y, spec.y_format) == (
        "grouped_bar",
        "region",
        ["net_revenue_previous", "net_revenue"],
        "currency",
    )


def test_suggests_lines_for_time_series() -> None:
    spec = suggest_chart(MONTHLY, "Trend")

    assert spec is not None and (spec.type, spec.x, spec.color) == ("line", "month", "region")


def test_no_chart_without_a_measure() -> None:
    assert suggest_chart(pd.DataFrame({"region": ["NA"]}), "t") is None


@pytest.mark.parametrize(
    ("spec", "problem"),
    [
        (ChartSpec(type="bar", title="t", x="planet", y=["net_revenue"]), "'planet' is not in"),
        (ChartSpec(type="bar", title="t", x="net_revenue", y=["region"]), "must be numeric"),
        (ChartSpec(type="line", title="t", x="region", y=["net_revenue"]), "needs a time x axis"),
        (ChartSpec(type="grouped_bar", title="t", x="region", y=["net_revenue"]), "several y"),
    ],
)
def test_validation(spec: ChartSpec, problem: str) -> None:
    assert any(problem in p for p in validate_chart(spec, QOQ))


def test_too_many_categories_for_bars() -> None:
    frame = pd.DataFrame({"c": [str(i) for i in range(40)], "v": range(40)})

    problems = validate_chart(ChartSpec(type="bar", title="t", x="c", y=["v"]), frame)

    assert "Too many categories" in problems[0]


def test_compiles_and_renders() -> None:
    spec = suggest_chart(QOQ, "QoQ")
    assert spec is not None

    vega = to_vega_lite(spec, QOQ)

    assert vega["mark"]["type"] == "bar"
    assert vega["encoding"]["xOffset"]["field"] == "series"
    assert render_png(vega).startswith(b"\x89PNG")
    assert "<svg" in render_svg(vega)


def test_invalid_specs_are_never_compiled() -> None:
    with pytest.raises(ValueError, match="Invalid chart"):
        to_vega_lite(ChartSpec(type="bar", title="t", x="nope", y=["net_revenue"]), QOQ)
