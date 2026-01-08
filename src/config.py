"""Configuration settings for the Physics QA system."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )

    # OpenAI API (for models not supported by OpenRouter)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
    )

    # Model Configuration
    generation_model: str = Field(
        default="anthropic/claude-opus-4", alias="GENERATION_MODEL"
    )
    judge_model: str = Field(default="anthropic/claude-sonnet-4", alias="JUDGE_MODEL")
    qwen_model: str = Field(default="qwen/qwen3-max", alias="QWEN_MODEL")

    # Cross-check models - stored as comma-separated string in env
    # Per spec: deepseek-v3.2, o4-mini-2025-04-16, gemini-3-pro, grok-4-0709
    crosscheck_models_str: str = Field(
        default="deepseek/deepseek-v3-0324,openai/o4-mini-2025-04-16,google/gemini-2.5-pro-preview-05-06,x-ai/grok-4",
        alias="CROSSCHECK_MODELS",
    )

    @property
    def crosscheck_models(self) -> List[str]:
        """Parse crosscheck models from comma-separated string."""
        return [m.strip() for m in self.crosscheck_models_str.split(",") if m.strip()]

    # Rate limiting and retries
    max_concurrent_requests: int = Field(default=50, alias="MAX_CONCURRENT_REQUESTS")
    requests_per_minute: int = Field(default=200, alias="REQUESTS_PER_MINUTE")
    max_retries: int = Field(default=10, alias="MAX_RETRIES")

    # Validation thresholds
    qwen_max_pass_rate: float = Field(default=0.05, alias="QWEN_MAX_PASS_RATE")
    min_crosscheck_models: int = Field(default=2, alias="MIN_CROSSCHECK_MODELS")
    min_crosscheck_total: int = Field(default=5, alias="MIN_CROSSCHECK_TOTAL")
    correctness_threshold: float = Field(default=0.90, alias="CORRECTNESS_THRESHOLD")

    # Generation parameters
    samples_per_qwen_check: int = Field(default=5, alias="SAMPLES_PER_QWEN_CHECK")
    samples_per_crosscheck: int = Field(default=5, alias="SAMPLES_PER_CROSSCHECK")
    target_count: int = Field(default=25, alias="TARGET_COUNT")

    # Output
    output_dir: str = Field(default="output", alias="OUTPUT_DIR")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        """Customize settings sources to handle comma-separated lists."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


def get_settings() -> Settings:
    """Get application settings, loading from .env file if present."""
    from dotenv import load_dotenv

    load_dotenv()
    return Settings()


# Singleton instance
_settings: Settings | None = None


def settings() -> Settings:
    """Get or create the settings singleton."""
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings
