"""Application configuration management for retail forecasting."""

from functools import lru_cache
import logging
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LOGGER = logging.getLogger(__name__)

_ALLOWED_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and optional .env file."""

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("ollama_host")
    @classmethod
    def validate_ollama_host(cls, value: str) -> str:
        """Ensure OLLAMA_HOST is a valid HTTP(S) URL string.

        Args:
            value: Raw environment value.

        Returns:
            Normalized host value.
        """
        normalized = value.strip().rstrip("/")
        if not normalized:
            raise ValueError("OLLAMA_HOST must not be empty")
        if not (normalized.startswith("http://") or normalized.startswith("https://")):
            raise ValueError("OLLAMA_HOST must start with http:// or https://")
        return normalized

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Ensure LOG_LEVEL is one of the recognized standard levels.

        Args:
            value: Raw environment value.

        Returns:
            Uppercase log level.
        """
        normalized = value.strip().upper()
        if normalized not in _ALLOWED_LOG_LEVELS:
            valid_levels = ", ".join(sorted(_ALLOWED_LOG_LEVELS))
            raise ValueError(f"LOG_LEVEL must be one of: {valid_levels}")
        return normalized


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache application settings.

    Returns:
        Cached Settings instance.
    """
    settings = Settings()
    LOGGER.debug("Settings loaded with model=%s", settings.ollama_model)
    return settings


def reset_settings_cache() -> None:
    """Clear cached settings for tests and controlled reloads."""
    get_settings.cache_clear()
