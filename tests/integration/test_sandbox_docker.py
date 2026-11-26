"""Container isolation. Needs Docker and `make sandbox`. Static checks are bypassed on purpose."""

from typing import Any

import pytest

from analytics_platform.config import Settings
from analytics_platform.tools.base import ErrorKind, ToolError
from analytics_platform.tools.sandbox.backends import DockerSandbox, SandboxOutput

pytestmark = pytest.mark.integration


def job(code: str) -> dict[str, Any]:
    return {"code": code, "datasets": {}, "max_output_chars": 2000}


async def run(code: str, **overrides: Any) -> SandboxOutput:
    return await DockerSandbox(Settings(_env_file=None, **overrides)).run(job(code))


async def test_runs_as_unprivileged_user_without_capabilities() -> None:
    out = await run(
        "import os\nstatus = open('/proc/self/status').read()\n"
        "result = {'uid': os.getuid(), 'caps': status.split('CapEff:')[1].split()[0]}"
    )

    assert out.result == {"type": "value", "value": {"uid": 10001, "caps": "0000000000000000"}}


async def test_no_network() -> None:
    out = await run("import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=3)\nresult=1")

    assert not out.ok and "Network is unreachable" in (out.error or "")


async def test_read_only_filesystem() -> None:
    out = await run("open('/opt/sandbox/x', 'w')\nresult = 1")

    assert not out.ok and "Read-only file system" in (out.error or "")


async def test_process_limit() -> None:
    out = await run("import os\nfor _ in range(200):\n    os.fork()\nresult = 1")

    assert not out.ok and out.error_type == "BlockingIOError"


async def test_memory_limit() -> None:
    out = await run("x = bytearray(2 * 1024**3)\nresult = 1")

    assert not out.ok and out.error_type == "MemoryError"


async def test_timeout_kills_the_container() -> None:
    with pytest.raises(ToolError) as info:
        await run("while True:\n    pass", sandbox_timeout_s=3)
    assert info.value.kind == ErrorKind.TIMEOUT


async def test_missing_image_is_reported_as_unavailable() -> None:
    with pytest.raises(ToolError) as info:
        await run("result = 1", sandbox_image="aeap-sandbox:does-not-exist")
    assert info.value.kind == ErrorKind.UNAVAILABLE
