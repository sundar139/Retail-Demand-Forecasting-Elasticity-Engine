"""Smoke tests for CLI commands."""

from typer.testing import CliRunner

from retail_forecasting.cli import app

RUNNER = CliRunner()


def test_cli_info_smoke() -> None:
    """The info command should execute and print package metadata."""
    result = RUNNER.invoke(app, ["info"])

    assert result.exit_code == 0
    assert "retail-forecasting-engine" in result.stdout


def test_cli_check_env_smoke() -> None:
    """The check-env command should pass with managed directories available."""
    result = RUNNER.invoke(app, ["check-env", "--create-missing"])

    assert result.exit_code == 0
    assert "Environment check" in result.stdout


def test_cli_show_paths_smoke() -> None:
    """The show-paths command should print resolved repository paths."""
    result = RUNNER.invoke(app, ["show-paths"])

    assert result.exit_code == 0
    assert "Resolved Paths" in result.stdout
    assert "data_raw_dir" in result.stdout
