from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class ErrorKind(StrEnum):
    TRANSIENT = "transient"  # retry the same call
    INVALID_INPUT = "invalid_input"  # fix the input (e.g. repair SQL) and retry
    PERMISSION = "permission"  # never retry; tell the user
    NOT_FOUND = "not_found"  # no data or no matching documents
    TIMEOUT = "timeout"  # retry once, or simplify the request
    UNAVAILABLE = "unavailable"  # dependency down; use a fallback source


class ToolError(Exception):
    """A classified tool failure, so the agent can choose retry, repair, fallback or report."""

    def __init__(
        self,
        message: str,
        *,
        kind: ErrorKind,
        details: str | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.details = details
        self.source = source  # the failing dependency, e.g. "finance", so a fallback can avoid it

    @property
    def retryable(self) -> bool:
        return self.kind in (ErrorKind.TRANSIENT, ErrorKind.TIMEOUT)


class Citation(BaseModel):
    """Where a piece of evidence came from, shown to the user next to the answer."""

    source_type: Literal["sql", "document", "web"]
    source_id: str
    title: str
    detail: str | None = None
    url: str | None = None
