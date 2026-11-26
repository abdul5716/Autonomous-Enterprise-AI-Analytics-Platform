"""Circuit breaker per dependency, so a database that is down is skipped instead of hammered."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from analytics_platform.tools.base import ErrorKind, ToolError


class CircuitOpenError(ToolError):
    def __init__(self, key: str, retry_in_s: float) -> None:
        super().__init__(
            f"{key} is temporarily disabled after repeated failures "
            f"(retrying in {retry_in_s:.0f}s).",
            kind=ErrorKind.UNAVAILABLE,
            source=key,
        )


@dataclass
class _Circuit:
    failures: int = 0
    opened_at: float | None = None
    probing: bool = False


@dataclass
class CircuitBreaker:
    """Opens after `threshold` consecutive failures; after `cooldown_s` lets one probe through.

    Shared across requests, so one request's discovery that Oracle is down spares the next.
    """

    threshold: int = 3
    cooldown_s: float = 60.0
    clock: Callable[[], float] = time.monotonic
    _circuits: dict[str, _Circuit] = field(default_factory=dict)

    def check(self, key: str) -> None:
        """Raise `CircuitOpenError` if calls to `key` should not be attempted now."""
        circuit = self._circuits.get(key)
        if circuit is None or circuit.opened_at is None:
            return
        waited = self.clock() - circuit.opened_at
        if waited < self.cooldown_s or circuit.probing:
            raise CircuitOpenError(key, max(self.cooldown_s - waited, 0))
        circuit.probing = True

    def is_open(self, key: str) -> bool:
        """Peek without consuming the half-open probe."""
        circuit = self._circuits.get(key)
        state = self.state(key)
        return state == "open" or (state == "half-open" and circuit is not None and circuit.probing)

    def record_success(self, key: str) -> None:
        self._circuits.pop(key, None)

    def record_failure(self, key: str) -> None:
        circuit = self._circuits.setdefault(key, _Circuit())
        circuit.failures += 1
        circuit.probing = False
        if circuit.failures >= self.threshold or circuit.opened_at is not None:
            circuit.opened_at = self.clock()

    def state(self, key: str) -> str:
        circuit = self._circuits.get(key)
        if circuit is None or circuit.opened_at is None:
            return "closed"
        return "half-open" if self.clock() - circuit.opened_at >= self.cooldown_s else "open"
