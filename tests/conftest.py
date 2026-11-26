import pytest

from analytics_platform.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        env="test",
        llm_fast_model="fast-model",
        llm_strong_model="strong-model",
        llm_judge_model="judge-model",
        llm_fallback_models=["fallback-model"],
        llm_max_retries=3,
        llm_retry_initial_wait_s=0,
    )
