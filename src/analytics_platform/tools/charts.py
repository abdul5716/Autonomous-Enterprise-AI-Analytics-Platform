"""Declarative charts: a small validated spec, compiled to Vega-Lite and rendered without a browser.

The LLM (or `suggest_chart`) only picks a chart type and columns; data always comes from tool
results, so a chart can't show numbers that were not computed.
"""

import re
from typing import Any, Literal

import altair as alt
import pandas as pd
import vl_convert as vlc
from pydantic import BaseModel, Field

ChartType = Literal["bar", "grouped_bar", "stacked_bar", "line", "area", "scatter"]

MAX_ROWS = 5_000
MAX_CATEGORIES = 30
MAX_SERIES = 12
TIME_LIKE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?|-Q[1-4])$")
FORMATS = {"currency": "$.3~s", "percent": ".1f", "number": ",.3~s"}


class ChartSpec(BaseModel):
    type: ChartType
    title: str
    x: str
    y: list[str] = Field(min_length=1, description="One or more numeric columns")
    color: str | None = Field(None, description="Categorical column that splits series")
    y_format: Literal["currency", "percent", "number"] = "number"
    x_title: str | None = None
    y_title: str | None = None
    sort: Literal["x", "-y", "none"] = "none"


def _is_time_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    values = series.dropna().astype(str)
    return bool(len(values)) and bool(values.map(lambda v: bool(TIME_LIKE.match(v))).all())


def validate_chart(spec: ChartSpec, frame: pd.DataFrame) -> list[str]:
    problems = []
    columns = set(frame.columns)
    for column in [spec.x, *spec.y, *([spec.color] if spec.color else [])]:
        if column not in columns:
            problems.append(f"Column '{column}' is not in the data ({', '.join(frame.columns)}).")
    if problems:
        return problems
    for column in spec.y:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            problems.append(f"y column '{column}' must be numeric.")
    if len(frame) > MAX_ROWS:
        problems.append(f"Too many rows to chart ({len(frame)} > {MAX_ROWS}); aggregate first.")
    if spec.type in ("line", "area") and not _is_time_like(frame[spec.x]):
        problems.append(f"A {spec.type} chart needs a time x axis; '{spec.x}' is not time-like.")
    if spec.type.endswith("bar") and frame[spec.x].nunique() > MAX_CATEGORIES:
        problems.append(f"Too many categories on '{spec.x}' for a bar chart (> {MAX_CATEGORIES}).")
    if spec.color and frame[spec.color].nunique() > MAX_SERIES:
        problems.append(f"Too many series in '{spec.color}' (> {MAX_SERIES}).")
    if len(spec.y) > 1 and spec.color:
        problems.append("Use either several y columns or a color column, not both.")
    if spec.type == "scatter" and len(spec.y) != 1:
        problems.append("A scatter chart takes exactly one y column.")
    if spec.type == "grouped_bar" and len(spec.y) == 1 and not spec.color:
        problems.append("A grouped bar chart needs several y columns or a color column.")
    return problems


def suggest_chart(frame: pd.DataFrame, title: str) -> ChartSpec | None:
    """A sensible default chart for a query result, or None if a table is better."""
    if frame.empty or len(frame) > MAX_ROWS:
        return None
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    measures = [c for c in numeric if not c.endswith("_change_pct")]
    dims = [c for c in frame.columns if c not in numeric]
    if not measures or not dims:
        return None
    y_format: Literal["currency", "percent", "number"] = (
        "currency"
        if any(k in measures[0] for k in ("revenue", "refund", "budget", "credit", "amount"))
        else "number"
    )
    time_dims = [d for d in dims if _is_time_like(frame[d])]
    if time_dims:
        others = [d for d in dims if d not in time_dims and frame[d].nunique() <= MAX_SERIES]
        spec = ChartSpec(
            type="line", title=title, x=time_dims[0], y=[measures[0]],
            color=others[0] if others else None, y_format=y_format, sort="x",
        )  # fmt: skip
    else:
        base = measures[0]
        compared = [c for c in measures if c == f"{base}_previous"]
        if compared:
            spec = ChartSpec(
                type="grouped_bar", title=title, x=dims[0], y=[compared[0], base],
                y_format=y_format, sort="-y",
            )  # fmt: skip
        elif len(dims) >= 2 and frame[dims[1]].nunique() <= MAX_SERIES:
            spec = ChartSpec(
                type="stacked_bar", title=title, x=dims[0], y=[base], color=dims[1],
                y_format=y_format,
            )  # fmt: skip
        else:
            spec = ChartSpec(
                type="bar", title=title, x=dims[0], y=[base], y_format=y_format, sort="-y"
            )
    return spec if not validate_chart(spec, frame) else None


def _label(column: str) -> str:
    return column.replace("_", " ").capitalize()


def to_vega_lite(spec: ChartSpec, frame: pd.DataFrame) -> dict[str, Any]:
    problems = validate_chart(spec, frame)
    if problems:
        raise ValueError("Invalid chart: " + " ".join(problems))
    data = frame.copy()
    time_x = _is_time_like(data[spec.x])
    if time_x:
        data[spec.x] = data[spec.x].astype(str)
    value, series = "value", "series"
    if len(spec.y) > 1:
        data = data.melt(id_vars=[spec.x], value_vars=spec.y, var_name=series, value_name=value)
        data[series] = data[series].map(_label)
        color_field: str | None = series
        y_field = value
    else:
        color_field = spec.color
        y_field = spec.y[0]
    x_sort: Any = None
    if spec.sort == "-y":
        x_sort = alt.EncodingSortField(field=y_field, op="sum", order="descending")
    elif spec.sort == "x" or time_x:
        x_sort = "ascending"
    y_title = spec.y_title or (_label(spec.y[0]) if len(spec.y) == 1 else "")
    x_enc = alt.X(
        f"{spec.x}:{'O' if time_x else 'N'}",
        sort=x_sort,
        title=spec.x_title or _label(spec.x),
        axis=alt.Axis(labelAngle=-45 if time_x else 0),
    )
    y_enc = alt.Y(f"{y_field}:Q", title=y_title, axis=alt.Axis(format=FORMATS[spec.y_format]))
    if spec.type == "scatter":
        x_enc = alt.X(f"{spec.x}:Q", title=spec.x_title or _label(spec.x))
    mark = {
        "bar": "bar", "grouped_bar": "bar", "stacked_bar": "bar",
        "line": "line", "area": "area", "scatter": "point",
    }[spec.type]  # fmt: skip
    chart = alt.Chart(data, title=spec.title)
    chart = chart.mark_line(point=True) if mark == "line" else getattr(chart, f"mark_{mark}")()
    encodings: dict[str, Any] = {"x": x_enc, "y": y_enc, "tooltip": list(data.columns)}
    if color_field:
        order = [_label(c) for c in spec.y] if len(spec.y) > 1 else None
        encodings["color"] = alt.Color(f"{color_field}:N", title=_label(color_field), sort=order)
        if spec.type == "grouped_bar":
            encodings["xOffset"] = alt.XOffset(f"{color_field}:N", sort=order)
    chart = chart.encode(**encodings).properties(width=520, height=300)
    return chart.to_dict()  # type: ignore[no-any-return]


def render_png(vega_lite: dict[str, Any], scale: float = 2.0) -> bytes:
    return vlc.vegalite_to_png(vega_lite, scale=scale)


def render_svg(vega_lite: dict[str, Any]) -> str:
    return vlc.vegalite_to_svg(vega_lite)
