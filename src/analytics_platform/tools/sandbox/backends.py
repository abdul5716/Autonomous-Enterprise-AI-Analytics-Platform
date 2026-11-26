"""Where analysis code runs: a locked-down Docker container, or (tests only) a subprocess."""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from analytics_platform.config import Settings
from analytics_platform.tools.base import ErrorKind, ToolError
from analytics_platform.tools.sandbox.runner import RESULT_MARKER

RUNNER = Path(__file__).with_name("runner.py")
OOM_EXIT_CODES = {137, -9}


class SandboxOutput(BaseModel):
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None
    stdout: str = ""
    duration_s: float = 0.0
    backend: str


class Sandbox(Protocol):
    name: str

    async def run(self, job: dict[str, Any]) -> SandboxOutput: ...


def parse_output(
    stdout: bytes, stderr: bytes, returncode: int | None, backend: str
) -> SandboxOutput:
    text = stdout.decode(errors="replace")
    marker = text.rfind(RESULT_MARKER)
    if marker == -1:
        if returncode in OOM_EXIT_CODES:
            return SandboxOutput(
                ok=False,
                error_type="MemoryError",
                error="The analysis ran out of memory; aggregate the data first.",
                backend=backend,
            )
        detail = stderr.decode(errors="replace").strip()[-500:]
        raise ToolError(
            f"Sandbox produced no result (exit code {returncode}): {detail}",
            kind=ErrorKind.UNAVAILABLE,
        )
    payload = json.loads(text[marker + len(RESULT_MARKER) :].strip().splitlines()[0])
    output = SandboxOutput(**payload, backend=backend)
    if output.error_type == "CPUTimeExceeded":
        raise ToolError(
            "Analysis used too much CPU time and was stopped; simplify it.",
            kind=ErrorKind.TIMEOUT,
        )
    return output


class DockerSandbox:
    name = "docker"

    def __init__(self, settings: Settings) -> None:
        self.image = settings.sandbox_image
        self.timeout_s = settings.sandbox_timeout_s
        self.memory = settings.sandbox_memory
        self.cpus = settings.sandbox_cpus

    def command(self, container: str) -> list[str]:
        return [
            "docker", "run", "-i", "--rm", "--name", container,
            "--network", "none",
            "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "10001:10001",
            "--memory", self.memory, "--memory-swap", self.memory,
            "--cpus", str(self.cpus), "--pids-limit", "64",
            "--ulimit", "nofile=256:256",
            "-e", "SANDBOX_RLIMITS=1", "-e", f"SANDBOX_CPU_S={int(self.timeout_s)}",
            self.image,
        ]  # fmt: skip

    async def run(self, job: dict[str, Any]) -> SandboxOutput:
        container = f"aeap-sandbox-{uuid.uuid4().hex[:12]}"
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.command(container),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise ToolError("Docker is not installed", kind=ErrorKind.UNAVAILABLE) from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(json.dumps(job).encode()), timeout=self.timeout_s
            )
        except TimeoutError as exc:
            await _kill(container)
            proc.kill()
            await proc.wait()
            raise ToolError(
                f"Analysis exceeded {self.timeout_s:.0f}s and was stopped; simplify it.",
                kind=ErrorKind.TIMEOUT,
            ) from exc
        err = stderr.decode(errors="replace")
        if proc.returncode == 125 or "Cannot connect to the Docker daemon" in err:
            hint = " Run `make sandbox` to build it." if "Unable to find image" in err else ""
            raise ToolError(
                f"Sandbox container could not start: {err.strip()[-300:]}{hint}",
                kind=ErrorKind.UNAVAILABLE,
            )
        return parse_output(stdout, stderr, proc.returncode, self.name)


async def _kill(container: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "docker", "kill", container,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )  # fmt: skip
    await proc.wait()


class SubprocessSandbox:
    """Same runner in a local subprocess with rlimits. NOT isolation: for tests and CI only."""

    name = "subprocess"

    def __init__(self, settings: Settings) -> None:
        self.timeout_s = settings.sandbox_timeout_s

    async def run(self, job: dict[str, Any]) -> SandboxOutput:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "SANDBOX_RLIMITS": "1",
            "SANDBOX_CPU_S": str(int(self.timeout_s)),
        }
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", str(RUNNER),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, env=env,
        )  # fmt: skip
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(json.dumps(job).encode()), timeout=self.timeout_s
            )
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise ToolError(
                f"Analysis exceeded {self.timeout_s:.0f}s and was stopped; simplify it.",
                kind=ErrorKind.TIMEOUT,
            ) from exc
        return parse_output(stdout, stderr, proc.returncode, self.name)


def build_sandbox(settings: Settings) -> Sandbox:
    backend: Literal["docker", "subprocess"] = settings.sandbox_backend
    return DockerSandbox(settings) if backend == "docker" else SubprocessSandbox(settings)
