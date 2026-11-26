"""Check connectivity to every backing service.

Usage: python -m analytics_platform.healthcheck [--llm]
"""

import argparse
import asyncio
import sys
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import oracledb
import psycopg
import redis

from analytics_platform.config import Settings, get_settings
from analytics_platform.llm import ModelTier, build_llm, track_usage


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_postgres(s: Settings) -> str:
    with psycopg.connect(s.postgres_dsn(), connect_timeout=3) as conn:
        row = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
    return f"pgvector {row[0]}" if row else "connected, pgvector missing"


def check_oracle(s: Settings) -> str:
    with oracledb.connect(
        user=s.oracle_user,
        password=s.oracle_password.get_secret_value(),
        dsn=s.oracle_dsn,
        tcp_connect_timeout=3,
    ) as conn:
        return f"Oracle {conn.version}"


def check_redis(s: Settings) -> str:
    client = redis.Redis.from_url(s.redis_url, socket_connect_timeout=2)
    client.ping()
    return f"Redis {client.info()['redis_version']}"


def check_phoenix(s: Settings) -> str:
    response = httpx.get(s.phoenix_url, timeout=3)
    response.raise_for_status()
    return f"HTTP {response.status_code}"


def check_llm(s: Settings) -> str:
    async def ping() -> str:
        with track_usage() as usage:
            reply = await build_llm(s).complete(
                [{"role": "user", "content": "Reply with the single word: pong"}],
                tier=ModelTier.FAST,
            )
        return f"{reply.model}: {reply.content.strip()!r} (${usage.cost_usd:.6f})"

    return asyncio.run(ping())


def run_checks(settings: Settings, *, include_llm: bool) -> list[CheckResult]:
    checks: dict[str, Callable[[Settings], str]] = {
        "postgres": check_postgres,
        "oracle": check_oracle,
        "redis": check_redis,
        "phoenix": check_phoenix,
    }
    if include_llm:
        checks["llm"] = check_llm

    results = []
    for name, check in checks.items():
        try:
            results.append(CheckResult(name, True, check(settings)))
        except Exception as exc:
            results.append(CheckResult(name, False, f"{type(exc).__name__}: {exc}"))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true", help="also make one real LLM call")
    args = parser.parse_args()

    results = run_checks(get_settings(), include_llm=args.llm)
    for r in results:
        print(f"{'OK  ' if r.ok else 'FAIL'}  {r.name:<9} {r.detail}")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
