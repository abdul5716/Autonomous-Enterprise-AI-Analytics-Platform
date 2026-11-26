"""Runs inside the sandbox. Reads one JSON job from stdin, writes one JSON result to stdout.

Job: {"code": str, "datasets": {name: {"columns": [...], "rows": [[...]], "dates": [...]}},
      "max_output_chars": int}
The code sees each dataset as a pandas DataFrame named after it (and in the `data` dict), and
must assign its answer to `result`. Standard library + pandas + numpy (+ scipy if installed).
"""

import contextlib
import io
import json
import math
import os
import sys
import time
import traceback
from typing import Any

RESULT_MARKER = "__SANDBOX_RESULT__"
CPU_LIMIT_MESSAGE = "CPU time limit exceeded"


class CPUTimeExceeded(BaseException):
    """BaseException so user code's `except Exception` can't swallow it."""


def _on_cpu_limit(signum: int, frame: Any) -> None:
    raise CPUTimeExceeded(CPU_LIMIT_MESSAGE)


def _limit_resources() -> None:
    if os.environ.get("SANDBOX_RLIMITS") != "1":
        return
    import resource
    import signal

    cpu_s = int(os.environ.get("SANDBOX_CPU_S", "30"))
    memory = int(os.environ.get("SANDBOX_MEMORY_BYTES", str(1024**3)))
    # SIGXCPU at the soft limit becomes a reportable error; the hard limit is a SIGKILL backstop.
    signal.signal(signal.SIGXCPU, _on_cpu_limit)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 2))
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item") and not isinstance(value, list | dict | str):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _serialise(result: Any) -> dict[str, Any]:
    import pandas as pd

    if isinstance(result, pd.Series):
        result = result.reset_index()
    if isinstance(result, pd.DataFrame):
        frame = result.reset_index() if not isinstance(result.index, pd.RangeIndex) else result
        return {
            "type": "table",
            "columns": [str(c) for c in frame.columns],
            "rows": [[_jsonable(v) for v in row] for row in frame.itertuples(index=False)],
        }
    if isinstance(result, dict):
        return {"type": "value", "value": {str(k): _jsonable(v) for k, v in result.items()}}
    if isinstance(result, list | tuple):
        return {"type": "value", "value": [_jsonable(v) for v in result]}
    return {"type": "value", "value": _jsonable(result)}


def main() -> None:
    _limit_resources()
    job = json.loads(sys.stdin.read())
    max_chars = int(job.get("max_output_chars", 20_000))
    started = time.perf_counter()
    out = io.StringIO()
    response: dict[str, Any]
    try:
        import numpy as np
        import pandas as pd

        data = {}
        for name, table in job["datasets"].items():
            frame = pd.DataFrame(table["rows"], columns=table["columns"])
            for column in table.get("dates", []):
                frame[column] = pd.to_datetime(frame[column])
            data[name] = frame
        namespace: dict[str, Any] = {"pd": pd, "np": np, "data": data, **data}
        with contextlib.redirect_stdout(out):
            exec(compile(job["code"], "<analysis>", "exec"), namespace)
        if "result" not in namespace:
            raise NameError("The code must assign its answer to a variable named `result`.")
        response = {"ok": True, "result": _serialise(namespace["result"])}
    except BaseException as exc:  # report every failure, including SystemExit and MemoryError
        frames = traceback.extract_tb(exc.__traceback__)
        user_frames = [f for f in frames if f.filename == "<analysis>"]
        line = user_frames[-1].lineno if user_frames else None
        response = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": f"{type(exc).__name__}: {exc}" + (f" (line {line})" if line else ""),
        }
    response["stdout"] = out.getvalue()[:max_chars]
    response["duration_s"] = round(time.perf_counter() - started, 3)
    encoded = json.dumps(response, default=str)
    if len(encoded) > max_chars * 5:
        encoded = json.dumps(
            {
                "ok": False,
                "error_type": "OutputTooLarge",
                "error": "Result is too large; aggregate or limit it before assigning `result`.",
                "stdout": "",
                "duration_s": response["duration_s"],
            }
        )
    real_stdout = sys.__stdout__
    assert real_stdout is not None
    real_stdout.write(RESULT_MARKER + encoded + "\n")


if __name__ == "__main__":
    main()
