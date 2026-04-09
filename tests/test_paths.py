"""Tests for project path resolution."""

from pathlib import Path

from retail_forecasting.config import Settings
from retail_forecasting.paths import build_project_paths, get_project_root, resolve_from_root


def test_resolve_from_root_handles_relative_and_absolute_paths(tmp_path: Path) -> None:
    """Path resolution should preserve absolute values and resolve relative values."""
    project_root = tmp_path / "repo"
    project_root.mkdir(parents=True, exist_ok=True)

    relative = resolve_from_root(project_root, Path("data/raw"))
    absolute_input = tmp_path / "external"
    absolute = resolve_from_root(project_root, absolute_input)

    assert relative == project_root / "data/raw"
    assert absolute == absolute_input


def test_build_project_paths_uses_settings_paths() -> None:
    """Configured path values should be resolved against project root."""
    settings = Settings(
        ollama_host="http://localhost:11434",
        ollama_model="llama3.2:latest",
        data_raw_dir=Path("custom/raw"),
        data_processed_dir=Path("custom/processed"),
        artifacts_dir=Path("custom/artifacts"),
        log_level="INFO",
    )

    paths = build_project_paths(settings)
    project_root = get_project_root()

    assert paths.project_root == project_root
    assert paths.data_raw_dir == project_root / "custom/raw"
    assert paths.data_processed_dir == project_root / "custom/processed"
    assert paths.artifacts_dir == project_root / "custom/artifacts"
    assert paths.data_interim_dir == project_root / "data/interim"
