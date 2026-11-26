"""Semantic layer: governed metric definitions compiled to warehouse SQL."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from analytics_platform.security import UserContext
from analytics_platform.tools.sql.periods import ResolvedPeriod, TimeRange


class FactDef(BaseModel):
    table: str
    alias: str
    joins: list[str]


class JoinDef(BaseModel):
    table: str
    alias: str
    condition: str


class MetricDef(BaseModel):
    fact: str
    label: str
    description: str
    sql: str
    format: Literal["currency", "integer", "number", "percent"]


class DimensionDef(BaseModel):
    description: str
    sql: str | dict[str, str]
    join: str | dict[str, str] | None = None
    values: list[str] | None = None
    time_grain: bool = False

    def sql_for(self, fact: str) -> str | None:
        return self.sql if isinstance(self.sql, str) else self.sql.get(fact)

    def join_for(self, fact: str) -> str | None:
        return self.join if isinstance(self.join, str) or self.join is None else self.join.get(fact)


class SemanticModel(BaseModel):
    version: int
    fiscal_year_start_month: int
    facts: dict[str, FactDef]
    joins: dict[str, JoinDef]
    time_column: str
    metrics: dict[str, MetricDef]
    dimensions: dict[str, DimensionDef]

    @classmethod
    def load(cls, path: str | Path) -> "SemanticModel":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def dimension_available(self, dimension: str, fact: str) -> bool:
        dim = self.dimensions[dimension]
        join = dim.join_for(fact)
        return dim.sql_for(fact) is not None and (join is None or join in self.facts[fact].joins)

    def catalog_text(self) -> str:
        lines = ["Metrics:"]
        for name, m in self.metrics.items():
            lines.append(f"- {name}: {m.label}. {m.description.strip()}")
        lines.append("Dimensions:")
        for name, d in self.dimensions.items():
            values = f" Values: {', '.join(d.values)}." if d.values else ""
            lines.append(f"- {name}: {d.description}{values}")
        for fact in self.facts:
            if any(m.fact == fact for m in self.metrics.values()) and fact != "revenue":
                usable = [d for d in self.dimensions if self.dimension_available(d, fact)]
                metrics = [n for n, m in self.metrics.items() if m.fact == fact]
                lines.append(f"Metrics {metrics} only support dimensions: {', '.join(usable)}.")
        return "\n".join(lines)


class Filter(BaseModel):
    dimension: str
    op: Literal["in", "not_in"] = "in"
    values: list[str | int] = Field(min_length=1)


class OrderBy(BaseModel):
    field: str
    descending: bool = True


class MetricQuery(BaseModel):
    metrics: list[str] = Field(min_length=1)
    dimensions: list[str] = []
    filters: list[Filter] = []
    time_range: TimeRange | None = None
    compare_to: Literal["previous_period", "same_period_last_year"] | None = None
    order_by: list[OrderBy] = []
    limit: int | None = Field(default=None, ge=1)


class CompiledQuery(BaseModel):
    sql: str
    params: list[Any]


class SemanticCompiler:
    def __init__(self, model: SemanticModel) -> None:
        self.model = model

    def validate(self, q: MetricQuery) -> list[str]:
        """Return human-readable problems; an empty list means the query can be compiled."""
        m = self.model
        errors = [f"Unknown metric '{name}'." for name in q.metrics if name not in m.metrics]
        names = [*q.dimensions, *(f.dimension for f in q.filters)]
        errors += [f"Unknown dimension '{d}'." for d in names if d not in m.dimensions]
        if errors:
            return errors

        for fact in {m.metrics[name].fact for name in q.metrics}:
            for d in names:
                if not m.dimension_available(d, fact):
                    metrics = [n for n in q.metrics if m.metrics[n].fact == fact]
                    errors.append(f"Dimension '{d}' is not available for metrics {metrics}.")
        for f in q.filters:
            allowed = m.dimensions[f.dimension].values
            if allowed is None:
                continue
            lookup = {v.lower(): v for v in allowed}
            bad = [v for v in f.values if str(v).lower() not in lookup]
            if bad:
                errors.append(
                    f"Invalid value(s) {bad} for '{f.dimension}'. Allowed: {', '.join(allowed)}."
                )
        outputs = {*q.metrics, *q.dimensions}
        errors += [
            f"Cannot order by '{o.field}': it is not a selected metric or dimension."
            for o in q.order_by
            if o.field not in outputs
        ]
        return errors

    def compile(
        self,
        q: MetricQuery,
        *,
        period: ResolvedPeriod | None,
        user: UserContext,
        limit: int,
    ) -> CompiledQuery:
        params: list[Any] = []
        by_fact: dict[str, list[str]] = {}
        for name in q.metrics:
            by_fact.setdefault(self.model.metrics[name].fact, []).append(name)
        selects = [
            self._fact_select(fact, metrics, q, period, user, params)
            for fact, metrics in by_fact.items()
        ]

        if len(selects) == 1:
            sql = selects[0]
        elif len(selects) == 2:
            ctes = f"WITH m0 AS (\n{selects[0]}\n),\nm1 AS (\n{selects[1]}\n)\n"
            if q.dimensions:
                dims = ", ".join(f"COALESCE(m0.{d}, m1.{d}) AS {d}" for d in q.dimensions)
                on = " AND ".join(f"m0.{d} IS NOT DISTINCT FROM m1.{d}" for d in q.dimensions)
                metric_cols = ", ".join(q.metrics)
                sql = ctes + f"SELECT {dims}, {metric_cols}\nFROM m0 FULL OUTER JOIN m1 ON {on}"
            else:
                sql = ctes + "SELECT * FROM m0 CROSS JOIN m1"
        else:
            raise ValueError("Metrics from more than two facts cannot be combined in one query.")

        sql += f"\nORDER BY {self._order_by(q)}\nLIMIT {int(limit)}"
        return CompiledQuery(sql=sql, params=params)

    def _fact_select(
        self,
        fact_name: str,
        metrics: list[str],
        q: MetricQuery,
        period: ResolvedPeriod | None,
        user: UserContext,
        params: list[Any],
    ) -> str:
        m = self.model
        fact = m.facts[fact_name]
        joins: set[str] = set()

        def dim_sql(name: str) -> str:
            dim = m.dimensions[name]
            join = dim.join_for(fact_name)
            if join:
                joins.add(join)
            sql = dim.sql_for(fact_name)
            assert sql is not None
            return sql

        columns = [f"{dim_sql(d)} AS {d}" for d in q.dimensions]
        columns += [f"{m.metrics[name].sql} AS {name}" for name in metrics]

        where: list[str] = []
        if period is not None:
            joins.add("date")
            where.append(f"{m.time_column} >= ? AND {m.time_column} < ?")
            params += [period.start, period.end]
        for f in q.filters:
            canonical = self._canonical_values(f)
            placeholders = ", ".join("?" for _ in canonical)
            operator = "IN" if f.op == "in" else "NOT IN"
            where.append(f"{dim_sql(f.dimension)} {operator} ({placeholders})")
            params += canonical
        if user.region_scope is not None:
            placeholders = ", ".join("?" for _ in user.region_scope)
            where.append(f"{dim_sql('region')} IN ({placeholders})")
            params += user.region_scope

        from_clause = f"FROM {fact.table} {fact.alias}"
        for join_name in sorted(joins):
            j = m.joins[join_name]
            from_clause += f"\nJOIN {j.table} {j.alias} ON {j.condition.format(fact=fact.alias)}"

        sql = f"SELECT {', '.join(columns)}\n{from_clause}"
        if where:
            sql += "\nWHERE " + "\n  AND ".join(where)
        if q.dimensions:
            sql += "\nGROUP BY " + ", ".join(str(i) for i in range(1, len(q.dimensions) + 1))
        return sql

    def _canonical_values(self, f: Filter) -> list[str | int]:
        allowed = self.model.dimensions[f.dimension].values
        if allowed is None:
            return list(f.values)
        lookup = {v.lower(): v for v in allowed}
        return [lookup[str(v).lower()] for v in f.values]

    def _order_by(self, q: MetricQuery) -> str:
        if q.order_by:
            return ", ".join(f"{o.field} {'DESC' if o.descending else 'ASC'}" for o in q.order_by)
        time_dims = [d for d in q.dimensions if self.model.dimensions[d].time_grain]
        if time_dims:
            return ", ".join(f"{d} ASC" for d in time_dims)
        return f"{q.metrics[0]} DESC NULLS LAST"
