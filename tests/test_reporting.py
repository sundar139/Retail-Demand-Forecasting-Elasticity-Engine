"""Tests for Phase 7 reporting, consistency reconciliation, and run manifest outputs."""

from pathlib import Path
import json

import pandas as pd
import pytest

from retail_forecasting.paths import ProjectPaths
from retail_forecasting.reporting import ReportingConfig, generate_reporting_artifacts


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


def _write_forecast_artifacts(paths: ProjectPaths) -> None:
    validation = pd.DataFrame(
        [
            {
                "split": "validation",
                "model_name": "lightgbm",
                "model_family": "lightgbm",
                "n_obs": 100,
                "mape": 20.0,
                "wmape": 5.5,
                "mae": 7.0,
                "rmse": 8.5,
            },
            {
                "split": "validation",
                "model_name": "xgboost",
                "model_family": "xgboost",
                "n_obs": 100,
                "mape": 21.0,
                "wmape": 6.0,
                "mae": 7.5,
                "rmse": 9.0,
            },
            {
                "split": "validation",
                "model_name": "naive_last",
                "model_family": "baseline",
                "n_obs": 100,
                "mape": 120.0,
                "wmape": 35.0,
                "mae": 20.0,
                "rmse": 25.0,
            },
        ]
    )
    test = pd.DataFrame(
        [
            {
                "split": "test",
                "model_name": "lightgbm",
                "model_family": "lightgbm",
                "n_obs": 80,
                "mape": 19.8,
                "wmape": 5.4,
                "mae": 6.9,
                "rmse": 8.3,
            },
            {
                "split": "test",
                "model_name": "xgboost",
                "model_family": "xgboost",
                "n_obs": 80,
                "mape": 20.7,
                "wmape": 5.9,
                "mae": 7.2,
                "rmse": 8.8,
            },
        ]
    )
    segment = pd.DataFrame(
        [
            {
                "split": "validation",
                "model_name": "lightgbm",
                "model_family": "lightgbm",
                "segment_key": "product_id=P1",
                "n_obs": 10,
                "mape": 18.0,
                "wmape": 5.0,
                "mae": 6.8,
                "rmse": 8.1,
            },
            {
                "split": "validation",
                "model_name": "xgboost",
                "model_family": "xgboost",
                "segment_key": "product_id=P1",
                "n_obs": 10,
                "mape": 19.0,
                "wmape": 5.3,
                "mae": 7.1,
                "rmse": 8.4,
            },
        ]
    )

    validation.to_csv(paths.artifacts_dir / "forecast_metrics_validation.csv", index=False)
    test.to_csv(paths.artifacts_dir / "forecast_metrics_test.csv", index=False)
    segment.to_csv(paths.artifacts_dir / "forecast_segment_metrics.csv", index=False)


def _write_training_registry(
    paths: ProjectPaths,
    *,
    lightgbm_llm_columns: str = "",
    lightgbm_llm_requested: bool = False,
) -> None:
    registry = pd.DataFrame(
        [
            {
                "trained_at_utc": "2026-04-08T10:00:00+00:00",
                "model_name": "lightgbm",
                "status": "success",
                "model_path": str(paths.artifacts_dir / "models" / "lightgbm_global.joblib"),
                "training_source": "split_feature_artifacts",
                "feature_count": 39,
                "llm_features_used": True,
                "llm_feature_columns": lightgbm_llm_columns,
                "llm_requested": lightgbm_llm_requested,
            },
            {
                "trained_at_utc": "2026-04-08T10:00:00+00:00",
                "model_name": "xgboost",
                "status": "success",
                "model_path": str(paths.artifacts_dir / "models" / "xgboost_global.joblib"),
                "training_source": "split_feature_artifacts",
                "feature_count": 39,
                "llm_features_used": False,
                "llm_feature_columns": "",
                "llm_requested": False,
            },
        ]
    )
    registry.to_csv(paths.artifacts_dir / "model_training_registry.csv", index=False)


def _write_conflicting_best_registry(paths: ProjectPaths) -> None:
    conflicting = pd.DataFrame(
        [
            {
                "selected_at_utc": "2026-04-08T10:00:00+00:00",
                "scope": "overall",
                "segment_key": "all",
                "optimize_metric": "wmape",
                "model_name": "xgboost",
                "model_family": "xgboost",
                "validation_mape": 21.0,
                "validation_wmape": 6.0,
                "validation_mae": 7.5,
                "validation_rmse": 9.0,
                "test_mape": 20.7,
                "test_wmape": 5.9,
                "test_mae": 7.2,
                "test_rmse": 8.8,
                "training_source": "split_feature_artifacts",
                "feature_count": 39,
                "llm_features_used": True,
                "llm_feature_columns": "",
                "model_artifact_path": str(paths.artifacts_dir / "models" / "xgboost_global.joblib"),
            }
        ]
    )
    conflicting.to_csv(paths.artifacts_dir / "best_model_registry.csv", index=False)


def _write_minimal_context_artifacts(
    paths: ProjectPaths,
    *,
    llm_requested: bool = False,
    llm_output_feature_count: int = 0,
    llm_output_feature_names: list[str] | None = None,
    planner_model_available: bool | None = None,
) -> None:
    output_names = llm_output_feature_names if llm_output_feature_names is not None else []
    (paths.artifacts_dir / "data_summary.json").write_text(
        json.dumps(
            {
                "source_filename": "retail.csv",
                "total_row_count": 1000,
                "min_date": "2024-01-01",
                "max_date": "2024-03-31",
            }
        ),
        encoding="utf-8",
    )
    (paths.artifacts_dir / "split_summary.json").write_text(
        json.dumps({"cutoffs": {"validation_start": "2024-03-01", "test_start": "2024-03-16"}}),
        encoding="utf-8",
    )
    (paths.artifacts_dir / "features_manual_summary.json").write_text(
        json.dumps({"feature_column_count": 39}),
        encoding="utf-8",
    )
    (paths.artifacts_dir / "llm_features_summary.json").write_text(
        json.dumps(
            {
                "llm_requested": llm_requested,
                "ollama_reachable": True,
                "planner_model_available": planner_model_available,
                "accepted_spec_count": 0,
                "output_feature_count": llm_output_feature_count,
                "output_feature_names": output_names,
                "output_paths": {
                    "features_llm_parquet": str(paths.data_processed_dir / "features_llm.parquet")
                },
            }
        ),
        encoding="utf-8",
    )


def _write_elasticity_artifacts(paths: ProjectPaths, *, warning_count: int) -> None:
    quality_status = "warning_inference_unstable" if warning_count > 0 else "ok"
    estimates = pd.DataFrame(
        [
            {
                "segment_key": "product_id=P1",
                "fit_status": "success",
                "quality_status": quality_status,
                "elasticity_estimate": -1.2,
                "lower_ci": -1.7,
                "upper_ci": -0.8,
                "sample_size": 120,
                "skip_reason": "",
                "warning_count": warning_count,
                "inference_warning_count": warning_count,
                "warnings_text": (
                    "Co-variance matrix is underdetermined. Inference will be invalid!"
                    if warning_count > 0
                    else ""
                ),
                "ci_caution_flag": warning_count > 0,
            }
        ]
    )
    estimates.to_csv(paths.artifacts_dir / "elasticity_estimates.csv", index=False)

    (paths.artifacts_dir / "elasticity_run_summary.json").write_text(
        json.dumps(
            {
                "total_segments_attempted": 1,
                "successful_fits": 1,
                "skipped_fits": 0,
                "failed_fits": 0,
                "inference_warning_count": warning_count,
                "inference_warnings_present": warning_count > 0,
                "ci_caution_present": warning_count > 0,
                "quality_status_counts": {quality_status: 1},
            }
        ),
        encoding="utf-8",
    )


def test_reporting_reconciles_best_model_with_metrics_source_of_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report generation should reconcile stale winner artifacts to metric source-of-truth."""
    from retail_forecasting import reporting

    paths = _build_temp_paths(tmp_path)
    _write_forecast_artifacts(paths)
    _write_training_registry(paths)
    _write_conflicting_best_registry(paths)
    _write_minimal_context_artifacts(paths)

    monkeypatch.setattr(reporting, "build_project_paths", lambda: paths)

    outputs = generate_reporting_artifacts(ReportingConfig(optimize_metric="wmape"))

    assert outputs["forecast_metrics_summary_csv"].exists()
    assert outputs["best_model_registry_csv"].exists()
    assert outputs["run_manifest_json"].exists()
    assert outputs["final_project_summary_json"].exists()
    assert outputs["final_run_report_markdown"].exists()

    best_registry = pd.read_csv(outputs["best_model_registry_csv"])
    overall = best_registry.loc[best_registry["scope"] == "overall"].iloc[0]

    assert str(overall["model_name"]) == "lightgbm"
    assert bool(overall["llm_features_used"]) is False


def test_reporting_handles_missing_optional_elasticity_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing elasticity artifacts should not block final report generation."""
    from retail_forecasting import reporting

    paths = _build_temp_paths(tmp_path)
    _write_forecast_artifacts(paths)
    _write_training_registry(paths)
    _write_minimal_context_artifacts(paths)

    monkeypatch.setattr(reporting, "build_project_paths", lambda: paths)

    outputs = generate_reporting_artifacts(ReportingConfig(optimize_metric="wmape"))

    elasticity_summary = pd.read_csv(outputs["elasticity_summary_csv"])
    assert elasticity_summary.empty

    manifest = json.loads(outputs["run_manifest_json"].read_text(encoding="utf-8"))
    warnings = manifest.get("warnings", [])
    assert isinstance(warnings, list)
    assert any("Elasticity estimates artifact is missing" in warning for warning in warnings)


def test_reporting_does_not_mark_llm_used_when_output_count_is_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM usage must remain false when summary reports zero validated output features."""
    from retail_forecasting import reporting

    paths = _build_temp_paths(tmp_path)
    _write_forecast_artifacts(paths)
    _write_training_registry(
        paths,
        lightgbm_llm_columns="llm_signal",
        lightgbm_llm_requested=True,
    )
    _write_minimal_context_artifacts(
        paths,
        llm_requested=True,
        llm_output_feature_count=0,
        llm_output_feature_names=[],
        planner_model_available=True,
    )

    # Deliberately create a stale features file; zero output count should still force "not used".
    pd.DataFrame(
        [{"date": "2024-01-01", "store_id": "S1", "product_id": "P1", "llm_signal": 1.0}]
    ).to_parquet(paths.data_processed_dir / "features_llm.parquet", index=False)

    monkeypatch.setattr(reporting, "build_project_paths", lambda: paths)
    outputs = generate_reporting_artifacts(ReportingConfig(optimize_metric="wmape"))

    best_registry = pd.read_csv(outputs["best_model_registry_csv"])
    overall = best_registry.loc[best_registry["scope"] == "overall"].iloc[0]
    assert bool(overall["llm_features_used"]) is False

    manifest = json.loads(outputs["run_manifest_json"].read_text(encoding="utf-8"))
    assert bool(manifest.get("llm_features_actually_used", True)) is False

    final_summary = json.loads(outputs["final_project_summary_json"].read_text(encoding="utf-8"))
    assert bool(final_summary.get("llm_features_actually_used", True)) is False


def test_reporting_acceptance_summary_matches_source_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance summary should stay consistent with manifest and final summary source values."""
    from retail_forecasting import reporting

    paths = _build_temp_paths(tmp_path)
    _write_forecast_artifacts(paths)
    _write_training_registry(
        paths,
        lightgbm_llm_columns="llm_signal",
        lightgbm_llm_requested=True,
    )
    _write_minimal_context_artifacts(
        paths,
        llm_requested=True,
        llm_output_feature_count=1,
        llm_output_feature_names=["llm_signal"],
        planner_model_available=True,
    )
    _write_elasticity_artifacts(paths, warning_count=1)

    pd.DataFrame(
        [{"date": "2024-01-01", "store_id": "S1", "product_id": "P1", "llm_signal": 1.0}]
    ).to_parquet(paths.data_processed_dir / "features_llm.parquet", index=False)

    monkeypatch.setattr(reporting, "build_project_paths", lambda: paths)
    outputs = generate_reporting_artifacts(ReportingConfig(optimize_metric="wmape"))

    best_registry = pd.read_csv(outputs["best_model_registry_csv"])
    overall = best_registry.loc[best_registry["scope"] == "overall"].iloc[0]
    assert bool(overall["llm_features_used"]) is True
    assert str(overall["llm_feature_columns"]) == "llm_signal"

    acceptance = json.loads(outputs["acceptance_summary_json"].read_text(encoding="utf-8"))
    manifest = json.loads(outputs["run_manifest_json"].read_text(encoding="utf-8"))
    final_summary = json.loads(outputs["final_project_summary_json"].read_text(encoding="utf-8"))

    assert bool(acceptance.get("llm_features_actually_used", False)) is True
    assert bool(manifest.get("llm_features_actually_used", False)) is True
    assert bool(final_summary.get("llm_features_actually_used", False)) is True

    assert bool(acceptance.get("elasticity_warning_presence", False)) is True
    assert bool(manifest.get("elasticity_inference_warnings_present", False)) is True
    assert bool(final_summary.get("elasticity_inference_warnings_present", False)) is True

    winner_payload = acceptance.get("forecasting_winner", {})
    assert isinstance(winner_payload, dict)
    winner = winner_payload.get("model_name")
    assert str(winner) == str(final_summary.get("best_model_name"))
