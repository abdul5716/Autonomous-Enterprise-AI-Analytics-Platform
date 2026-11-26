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
    postgres_rag_user: str = "rag_reader"
    postgres_rag_password: SecretStr = SecretStr("rag_reader_dev")

    oracle_host: str = "localhost"
    oracle_port: int = 51521
    oracle_service: str = "FREEPDB1"
    oracle_user: str = "finance"
    oracle_password: SecretStr = SecretStr("finance_dev")
    oracle_admin_user: str = "system"
    oracle_admin_password: SecretStr = SecretStr("oracle_admin_dev")
    oracle_ro_user: str = "finance_ro"
    oracle_ro_password: SecretStr = SecretStr("finance_ro_dev")

    redis_url: str = "redis://localhost:56379/0"
    duckdb_path: str = "data/warehouse/nimbus.duckdb"
    semantic_layer_path: str = "semantic_layer/metrics.yaml"
    sql_max_rows: int = 1000
    sql_timeout_s: float = 15.0
    documents_dir: str = "data/documents"
    # Expected answers for evals. Never exposed to the agent's tools.
    answer_key_path: str = "data/generated/answer_key.json"
    phoenix_url: str = "http://localhost:6006"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    model_cache_dir: str = "data/models"
    rag_candidates: int = 20  # per retriever, before fusion and reranking
    rag_top_k: int = 5
    # Reranker relevance bands, calibrated on the demo corpus: unrelated questions score
    # below 0.001, vague but answerable ones 0.02-0.5, direct matches above 0.5.
    rag_min_relevance: float = 0.01  # below: no evidence, the LLM is not called
    rag_strong_relevance: float = 0.5  # below: weak evidence, the answer is flagged

    # auto: live search, falling back to the offline fixtures; fixtures: offline only (evals, CI)
    web_mode: Literal["auto", "live", "fixtures"] = "auto"
    web_fixtures_dir: str = "data/web_fixtures"
    web_search_backends: list[str] = ["bing", "yahoo", "duckduckgo"]  # tried in order (ddgs)
    web_max_results: int = 5  # per query
    web_max_pages: int = 6  # fetched per research call
    web_fetch_timeout_s: float = 10.0
    web_max_page_bytes: int = 3_000_000
    web_max_page_chars: int = 6_000  # extracted text kept per page
    web_min_source_score: float = 0.35
    web_cache_ttl_s: int = 24 * 3600

    def postgres_dsn(self, *, read_only: bool = False) -> str:
        user, password = (
            (self.postgres_ro_user, self.postgres_ro_password)
            if read_only
            else (self.postgres_user, self.postgres_password)
        )
        return self._postgres_url(user, password)

    def postgres_rag_dsn(self) -> str:
        return self._postgres_url(self.postgres_rag_user, self.postgres_rag_password)

    def _postgres_url(self, user: str, password: SecretStr) -> str:
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
