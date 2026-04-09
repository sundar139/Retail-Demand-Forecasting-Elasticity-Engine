"""Tests for Phase 7 full-pipeline orchestration."""

from pathlib import Path
import json

import pytest

from retail_forecasting.paths import ProjectPaths
from retail_forecasting.pipeline import FullPipelineConfig, run_full_pipeline


def _build_temp_paths(tmp_path: Path) -> ProjectPaths:
    data_raw = tmp_path / "data" / "raw"
    data_interim = tmp_path / "data" / "interim"
    data_processed = tmp_path / "data" / "processed"
    artifacts = tmp_path / "artifacts"
    notebooks = tmp_path / "notebooks"
    reports = tmp_path / "reports" / "figures"
    prompts = tmp_path / "prompts"
    scripts = tmp_path / "scripts"

    for directory in [
        data_raw,
        data_interim,
        data_processed,
        artifacts,
        notebooks,
        reports,
        prompts,
        scripts,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        project_root=tmp_path,
        data_raw_dir=data_raw,
        data_interim_dir=data_interim,
        data_processed_dir=data_processed,
        artifacts_dir=artifacts,
        notebooks_dir=notebooks,
        reports_figures_dir=reports,
        prompts_dir=prompts,
        scripts_dir=scripts,
    )


def test_run_full_pipeline_respects_skip_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skipped stages should not execute while required stages continue."""
    from retail_forecasting import pipeline

    paths = _build_temp_paths(tmp_path)
    monkeypatch.setattr(pipeline, "build_project_paths", lambda: paths)

    calls: list[str] = []

    def stage(name: str):
        def _runner(*args, **kwargs):
            calls.append(name)
            artifact = paths.artifacts_dir / f"{name}.json"
            artifact.write_text("{}", encoding="utf-8")
            return {f"{name}_artifact": artifact}

        return _runner

    monkeypatch.setattr(pipeline, "_validate_data_stage", stage("validate_data"))
    monkeypatch.setattr(pipeline, "prepare_data_pipeline", stage("prepare_data"))
    monkeypatch.setattr(pipeline, "build_manual_features_pipeline", stage("build_manual_features"))
    monkeypatch.setattr(pipeline, "plan_llm_features_pipeline", stage("plan_llm_features"))
    monkeypatch.setattr(pipeline, "build_llm_features_pipeline", stage("build_llm_features"))
    monkeypatch.setattr(pipeline, "fit_elasticity_pipeline", stage("fit_elasticity"))
    monkeypatch.setattr(pipeline, "run_baseline_benchmark_pipeline", stage("run_baselines"))
    monkeypatch.setattr(pipeline, "tune_forecast_models_pipeline", stage("tune_forecast_models"))
    monkeypatch.setattr(pipeline, "train_forecast_models_pipeline", stage("train_forecast_models"))
    monkeypatch.setattr(pipeline, "evaluate_forecast_models_pipeline", stage("evaluate_forecast_models"))
    monkeypatch.setattr(pipeline, "generate_reporting_artifacts", stage("generate_report"))

    outputs = run_full_pipeline(
        FullPipelineConfig(
            skip_llm=True,
            skip_elasticity=True,
            skip_tuning=True,
        )
    )

    assert "plan_llm_features" not in calls
    assert "build_llm_features" not in calls
    assert "fit_elasticity" not in calls
    assert "tune_forecast_models" not in calls

    assert "run_baselines" in calls
    assert "train_forecast_models" in calls
    assert "evaluate_forecast_models" in calls
    assert "generate_report" in calls

    assert any(key.endswith("generate_report_artifact") for key in outputs)


def test_run_full_pipeline_writes_failure_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failures should raise clearly and still produce a failed run manifest."""
    from retail_forecasting import pipeline

    paths = _build_temp_paths(tmp_path)
    monkeypatch.setattr(pipeline, "build_project_paths", lambda: paths)

    def validate_stage(input_path):
        return {"source_csv": paths.data_raw_dir / "source.csv"}

    def broken_prepare(*args, **kwargs):
        raise ValueError("intentional failure")

    monkeypatch.setattr(pipeline, "_validate_data_stage", validate_stage)
    monkeypatch.setattr(pipeline, "prepare_data_pipeline", broken_prepare)

    with pytest.raises(RuntimeError, match="run-full-pipeline failed"):
        run_full_pipeline(FullPipelineConfig())

    manifest_path = paths.artifacts_dir / "run_manifest.json"
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload.get("status") == "failed"
    stage_records = payload.get("stage_records", [])
    assert isinstance(stage_records, list)
    assert any(record.get("status") == "failed" for record in stage_records)
