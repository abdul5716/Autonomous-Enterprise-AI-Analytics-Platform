from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"

    llm_provider: Literal["litellm", "fake"] = "litellm"
    llm_fast_model: str = "gpt-4o-mini"
    llm_strong_model: str = "gpt-4o-mini"
    llm_judge_model: str = "gpt-4o-mini"
    llm_fallback_models: list[str] = []
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3
    llm_retry_initial_wait_s: float = 1.0
    llm_cache_enabled: bool = True
    llm_cache_ttl_s: int = 7 * 24 * 3600

    tavily_api_key: SecretStr | None = None

    postgres_host: str = "localhost"
    postgres_port: int = 55432
    postgres_db: str = "nimbus"
    postgres_user: str = "nimbus"
    postgres_password: SecretStr = SecretStr("nimbus_dev")
    postgres_ro_user: str = "analyst_ro"
    postgres_ro_password: SecretStr = SecretStr("analyst_ro_dev")

    oracle_host: str = "localhost"
    oracle_port: int = 51521
    oracle_service: str = "FREEPDB1"
    oracle_user: str = "finance"
    oracle_password: SecretStr = SecretStr("finance_dev")

    redis_url: str = "redis://localhost:56379/0"
    duckdb_path: str = "data/warehouse/nimbus.duckdb"
    documents_dir: str = "data/documents"
    # Expected answers for evals. Never exposed to the agent's tools.
    answer_key_path: str = "data/generated/answer_key.json"
    phoenix_url: str = "http://localhost:6006"

    def postgres_dsn(self, *, read_only: bool = False) -> str:
        user, password = (
            (self.postgres_ro_user, self.postgres_ro_password)
            if read_only
            else (self.postgres_user, self.postgres_password)
        )
        return (
            f"postgresql://{user}:{password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def oracle_dsn(self) -> str:
        return f"{self.oracle_host}:{self.oracle_port}/{self.oracle_service}"


@lru_cache
def get_settings() -> Settings:
    # Provider keys (OPENAI_API_KEY, GEMINI_API_KEY, ...) are read by LiteLLM from os.environ.
    load_dotenv(override=False)
    return Settings()
