"""Tests for environment-backed settings."""

from pathlib import Path

from retail_forecasting.config import get_settings, reset_settings_cache


def test_settings_load_from_environment(monkeypatch) -> None:
    """Settings should load and normalize values from environment variables."""
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434/")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    monkeypatch.setenv("DATA_RAW_DIR", "data/raw")
    monkeypatch.setenv("DATA_PROCESSED_DIR", "data/processed")
    monkeypatch.setenv("ARTIFACTS_DIR", "artifacts")
    monkeypatch.setenv("LOG_LEVEL", "debug")

    reset_settings_cache()
    settings = get_settings()

    assert settings.ollama_host == "http://localhost:11434"
    assert settings.ollama_model == "llama3.2:3b"
    assert settings.data_raw_dir == Path("data/raw")
    assert settings.data_processed_dir == Path("data/processed")
    assert settings.artifacts_dir == Path("artifacts")
    assert settings.log_level == "DEBUG"

    reset_settings_cache()
