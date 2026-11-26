"""Python analysis over query results: LLM-written pandas code, checked, sandboxed, repaired."""

import datetime as dt
import math
from typing import Any

import pandas as pd
from pydantic import BaseModel

from analytics_platform.config import Settings
from analytics_platform.llm import BaseLLM, Message, ModelTier
from analytics_platform.security import UserContext
from analytics_platform.tools.base import ErrorKind, ToolError
from analytics_platform.tools.sandbox.backends import Sandbox, SandboxOutput, build_sandbox
from analytics_platform.tools.sandbox.checks import ALLOWED_IMPORTS, check_code

MAX_ATTEMPTS = 3

ANALYSIS_SYSTEM = """\
Write Python (pandas/numpy) code that answers the analyst's question from these DataFrames,
which are already loaded as variables:

{datasets}

Rules:
- Assign the final answer to `result`: a DataFrame (preferred), a dict of named values, or a number.
- Keep `result` small (aggregated, at most a few hundred rows) with clear column names.
- Allowed imports: {imports}. No file, network or system access.
- Compute from the data only; never type numbers in by hand.
"""


class AnalysisCode(BaseModel):
    code: str
    explanation: str


class AnalysisResult(BaseModel):
    question: str
    code: str
    explanation: str
    output: SandboxOutput
    attempts: int
    inputs: list[str]

    def to_frame(self) -> pd.DataFrame | None:
        result = self.output.result
        if not result or result.get("type") != "table":
            return None
        return pd.DataFrame(result["rows"], columns=result["columns"])


def _cell(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp | dt.date | dt.datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def frame_payload(frame: pd.DataFrame) -> dict[str, Any]:
    dates = [
        str(c)
        for c in frame.columns
        if pd.api.types.is_datetime64_any_dtype(frame[c])
        or (
            len(frame)
            and isinstance(frame[c].dropna().iloc[0] if frame[c].notna().any() else None, dt.date)
        )
    ]
    return {
        "columns": [str(c) for c in frame.columns],
        "rows": [[_cell(v) for v in row] for row in frame.itertuples(index=False)],
        "dates": dates,
    }


def describe(datasets: dict[str, pd.DataFrame]) -> str:
    lines = []
    for name, frame in datasets.items():
        columns = ", ".join(f"{c} ({frame[c].dtype})" for c in frame.columns)
        lines.append(f"`{name}`: {len(frame)} rows; columns: {columns}")
        lines.append(frame.head(3).to_string(index=False, max_colwidth=30))
    return "\n".join(lines)


class AnalysisTool:
    def __init__(self, llm: BaseLLM, settings: Settings, sandbox: Sandbox | None = None) -> None:
        self.llm = llm
        self.sandbox = sandbox or build_sandbox(settings)
        self.max_rows = settings.sandbox_max_input_rows
        self.max_output_chars = settings.sandbox_max_output_chars

    async def run_code(self, code: str, datasets: dict[str, pd.DataFrame]) -> SandboxOutput:
        problems = check_code(code)
        if problems:
            raise ToolError(
                "Code rejected: " + " ".join(problems), kind=ErrorKind.INVALID_INPUT, details=code
            )
        for name, frame in datasets.items():
            if not name.isidentifier():
                raise ToolError(
                    f"Dataset name {name!r} is not an identifier", kind=ErrorKind.INVALID_INPUT
                )
            if len(frame) > self.max_rows:
                raise ToolError(
                    f"Dataset {name!r} has {len(frame)} rows (limit {self.max_rows}); "
                    "aggregate it in SQL first.",
                    kind=ErrorKind.INVALID_INPUT,
                )
        job = {
            "code": code,
            "datasets": {n: frame_payload(f) for n, f in datasets.items()},
            "max_output_chars": self.max_output_chars,
        }
        return await self.sandbox.run(job)

    async def analyze(
        self, question: str, datasets: dict[str, pd.DataFrame], user: UserContext
    ) -> AnalysisResult:
        """Generate code, run it, and feed check or runtime errors back (up to 3 attempts)."""
        messages: list[Message] = [
            {
                "role": "system",
                "content": ANALYSIS_SYSTEM.format(
                    datasets=describe(datasets), imports=", ".join(sorted(ALLOWED_IMPORTS))
                ),
            },
            {"role": "user", "content": question},
        ]
        last_error = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            draft = await self.llm.structured(messages, AnalysisCode, tier=ModelTier.STRONG)
            try:
                output = await self.run_code(draft.code, datasets)
            except ToolError as exc:
                if exc.kind != ErrorKind.INVALID_INPUT or exc.details != draft.code:
                    raise
                last_error = str(exc)
            else:
                if output.ok:
                    return AnalysisResult(
                        question=question,
                        code=draft.code,
                        explanation=draft.explanation,
                        output=output,
                        attempts=attempt,
                        inputs=list(datasets),
                    )
                last_error = output.error or "unknown error"
            messages += [
                {"role": "assistant", "content": draft.model_dump_json()},
                {"role": "user", "content": f"That failed:\n{last_error}\nReturn fixed code."},
            ]
        raise ToolError(
            f"Analysis failed after {MAX_ATTEMPTS} attempts: {last_error}",
            kind=ErrorKind.INVALID_INPUT,
        )
