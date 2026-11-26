import pytest

from analytics_platform.config import Settings


def test_env_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_STRONG_MODEL", "gemini/gemini-2.0-flash")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", '["gpt-4o-mini"]')
    monkeypatch.setenv("POSTGRES_PORT", "6543")

    s = Settings(_env_file=None)

    assert s.llm_strong_model == "gemini/gemini-2.0-flash"
    assert s.llm_fallback_models == ["gpt-4o-mini"]
    assert s.postgres_port == 6543


def test_read_only_dsn_uses_analyst_role() -> None:
    s = Settings(_env_file=None)

    assert s.postgres_dsn(read_only=True).startswith("postgresql://analyst_ro:")
    assert s.postgres_dsn().startswith("postgresql://nimbus:")


def test_secrets_are_masked_in_repr() -> None:
    s = Settings(_env_file=None)

    assert "nimbus_dev" not in repr(s)
