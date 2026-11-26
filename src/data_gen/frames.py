from collections.abc import Iterator
from typing import Any

import pandas as pd


def rows(df: pd.DataFrame, columns: list[str]) -> Iterator[tuple[Any, ...]]:
    """Yield plain-Python tuples with pandas missing values converted to None."""
    subset = df[columns].astype(object)
    for row in subset.itertuples(index=False, name=None):
        yield tuple(None if _is_missing(v) else v for v in row)


def _is_missing(value: Any) -> bool:
    return value is None or (not isinstance(value, str) and bool(pd.isna(value)))
