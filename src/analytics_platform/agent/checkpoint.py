"""Checkpoints in Postgres, so a paused (clarifying) or interrupted run can resume later."""

import enum
import inspect
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any, get_args, get_type_hints
from urllib.parse import quote

import psycopg
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from analytics_platform.agent.models import AgentState
from analytics_platform.config import Settings

SCHEMA = "agent"


def _model_types(annotation: Any) -> Iterator[type]:
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel | enum.Enum):
        yield annotation
    for arg in get_args(annotation):
        yield from _model_types(arg)


def state_types() -> set[type]:
    """Every pydantic model and enum that can appear in agent state, found by walking fields."""
    hints = get_type_hints(AgentState, include_extras=True)
    queue = [t for hint in hints.values() for t in _model_types(hint)]
    found: set[type] = set()
    while queue:
        cls = queue.pop()
        if cls in found:
            continue
        found.add(cls)
        if issubclass(cls, BaseModel):
            for field in cls.model_fields.values():
                queue.extend(_model_types(field.annotation))
    return found


def serializer() -> JsonPlusSerializer:
    """Only the agent's own state types may be deserialised from a checkpoint."""
    allowed = [(t.__module__, t.__name__) for t in state_types()]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def memory_checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=serializer())


def checkpoint_dsn(settings: Settings) -> str:
    return settings.postgres_dsn() + "?options=" + quote(f"-c search_path={SCHEMA}")


@asynccontextmanager
async def postgres_checkpointer(settings: Settings) -> AsyncIterator[AsyncPostgresSaver]:
    """Checkpoint tables live in their own `agent` schema, apart from the business data."""
    async with await psycopg.AsyncConnection.connect(
        settings.postgres_dsn(), autocommit=True
    ) as conn:
        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    async with AsyncPostgresSaver.from_conn_string(
        checkpoint_dsn(settings), serde=serializer()
    ) as saver:
        await saver.setup()
        yield saver
