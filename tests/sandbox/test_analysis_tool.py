import datetime as dt

import pandas as pd
import pytest

from analytics_platform.config import Settings
from analytics_platform.llm import FakeLLM
from analytics_platform.security import DEMO_USERS
from analytics_platform.tools.base import ErrorKind, ToolError
from analytics_platform.tools.sandbox.backends import SubprocessSandbox
from analytics_platform.tools.sandbox.tool import AnalysisTool, describe, frame_payload

USER = DEMO_USERS["finance.admin"]
REVENUE = pd.DataFrame(
    {
        "month": [
            dt.date(2026, 6, 1),
            dt.date(2026, 7, 1),
            dt.date(2026, 6, 1),
            dt.date(2026, 7, 1),
        ],
        "region": ["EMEA", "EMEA", "NA", "NA"],
        "net_revenue": [100.0, 70.0, 200.0, 196.0],
    }
)


def make_tool(llm: FakeLLM | None = None, **overrides: object) -> AnalysisTool:
    settings = Settings(_env_file=None, sandbox_backend="subprocess", **overrides)  # type: ignore[arg-type]
    return AnalysisTool(llm or FakeLLM(), settings, sandbox=SubprocessSandbox(settings))


def code(source: str) -> dict[str, str]:
    return {"code": source, "explanation": "test"}


async def test_runs_pandas_and_returns_a_table() -> None:
    source = (
        "pivot = revenue.pivot_table(index='region', columns=revenue.month.dt.month, "
        "values='net_revenue')\n"
        "result = ((pivot[7] / pivot[6] - 1) * 100).round(1).rename('change_pct')"
    )

    out = await make_tool().run_code(source, {"revenue": REVENUE})

    assert out.ok, out.error
    assert out.result == {
        "type": "table",
        "columns": ["region", "change_pct"],
        "rows": [["EMEA", -30.0], ["NA", -2.0]],
    }


async def test_scalars_dicts_and_stdout() -> None:
    source = (
        "print('rows', len(revenue))\n"
        "result = {'total': revenue.net_revenue.sum(), 'n': len(revenue)}"
    )

    out = await make_tool().run_code(source, {"revenue": REVENUE})

    assert out.result == {"type": "value", "value": {"total": 566.0, "n": 4}}
    assert out.stdout == "rows 4\n"


async def test_runtime_errors_report_the_line() -> None:
    out = await make_tool().run_code("x = 1\nresult = revenue['missing']", {"revenue": REVENUE})

    assert not out.ok
    assert out.error == "KeyError: 'missing' (line 2)"


async def test_unsafe_code_never_runs() -> None:
    with pytest.raises(ToolError, match="import of 'os'") as info:
        await make_tool().run_code("import os\nresult = os.listdir('/')", {})
    assert info.value.kind == ErrorKind.INVALID_INPUT


async def test_large_inputs_are_refused() -> None:
    with pytest.raises(ToolError, match="aggregate it in SQL first"):
        await make_tool(sandbox_max_input_rows=2).run_code("result = 1", {"revenue": REVENUE})


async def test_runaway_code_is_stopped() -> None:
    with pytest.raises(ToolError) as info:
        await make_tool(sandbox_timeout_s=1).run_code("while True:\n    pass\nresult = 1", {})
    assert info.value.kind == ErrorKind.TIMEOUT


async def test_analyze_repairs_failing_code() -> None:
    llm = (
        FakeLLM()
        .when(
            "That failed:.*KeyError", code("result = revenue.groupby('region').net_revenue.sum()")
        )
        .when("That failed:.*import of 'os'", code("result = revenue['revenue']"))
        .when("Write Python", code("import os\nresult = 1"))
    )

    r = await make_tool(llm).analyze("Revenue by region", {"revenue": REVENUE}, USER)

    assert r.attempts == 3
    frame = r.to_frame()
    assert frame is not None and frame.to_dict("list") == {
        "region": ["EMEA", "NA"],
        "net_revenue": [170.0, 396.0],
    }


async def test_analyze_gives_up_after_three_attempts() -> None:
    llm = FakeLLM().when("Write Python", code("result = revenue['nope']"))

    with pytest.raises(ToolError, match="failed after 3 attempts: KeyError"):
        await make_tool(llm).analyze("x", {"revenue": REVENUE}, USER)


def test_prompt_describes_datasets_and_dates_survive_transport() -> None:
    text = describe({"revenue": REVENUE})
    payload = frame_payload(REVENUE)

    assert "`revenue`: 4 rows; columns: month (object), region" in text
    assert payload["dates"] == ["month"] and payload["rows"][0][0] == "2026-06-01"
