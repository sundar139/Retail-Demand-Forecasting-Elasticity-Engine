"""Tests for Phase 7 prediction and export workflows."""

from pathlib import Path

import joblib
import pandas as pd
import pytest

from retail_forecasting.forecasting_models import TrainedForecastModelArtifact
from retail_forecasting.paths import ProjectPaths
from retail_forecasting.predict import (
    ForecastExportConfig,
    ForecastNextConfig,
    export_forecasts_pipeline,
    forecast_next_pipeline,
)


class DummyEstimator:
    """Simple deterministic estimator for artifact prediction tests."""

    def predict(self, features: pd.DataFrame):
        values = pd.to_numeric(features["units_sold_lag_1"], errors="coerce").fillna(0.0)
        return (values + 1.0).to_numpy(dtype="float64")


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


def _write_feature_split_artifacts(paths: ProjectPaths) -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    rows: list[dict[str, object]] = []

    for day_idx, date_value in enumerate(dates, start=1):
        rows.append(
            {
                "date": date_value,
                "store_id": "S1",
                "product_id": "P1",
                "units_sold": float(10 + day_idx),
                "price": 8.0,
                "units_sold_lag_1": float(9 + day_idx) if day_idx > 1 else 10.0,
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_parquet(paths.data_processed_dir / "features_manual.parquet", index=False)
    frame.iloc[:20].to_parquet(paths.data_processed_dir / "features_train.parquet", index=False)
    frame.iloc[20:25].to_parquet(paths.data_processed_dir / "features_validation.parquet", index=False)
    frame.iloc[25:].to_parquet(paths.data_processed_dir / "features_test.parquet", index=False)


def _write_best_model_artifact(paths: ProjectPaths) -> None:
    model_path = paths.artifacts_dir / "models" / "lightgbm_global.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = TrainedForecastModelArtifact(
        model_name="lightgbm",
        model_family="lightgbm",
        segment_mode="global",
        estimator=DummyEstimator(),
        estimators_by_product=None,
        fallback_estimator=None,
        feature_columns=["units_sold_lag_1"],
        feature_medians={"units_sold_lag_1": 0.0},
        category_maps={},
        target_column="units_sold",
        training_source="split_feature_artifacts",
        llm_features_used=False,
        llm_feature_columns_in_matrix=[],
        random_state=42,
    )
    joblib.dump(artifact, model_path)

    registry = pd.DataFrame(
        [
            {
                "scope": "overall",
                "model_name": "lightgbm",
                "model_family": "lightgbm",
                "model_artifact_path": str(model_path),
            }
        ]
    )
    registry.to_csv(paths.artifacts_dir / "best_model_registry.csv", index=False)


def test_forecast_next_generates_latest_forecast_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """forecast-next should produce aligned predictions with identifiers and date columns."""
    from retail_forecasting import predict

    paths = _build_temp_paths(tmp_path)
    _write_feature_split_artifacts(paths)
    _write_best_model_artifact(paths)

    monkeypatch.setattr(predict, "build_project_paths", lambda: paths)

    outputs = forecast_next_pipeline(ForecastNextConfig(use_llm_features=False))

    forecast_path = outputs["forecasts_latest_csv"]
    assert forecast_path.exists()

    frame = pd.read_csv(forecast_path)
    required = {"date", "store_id", "product_id", "actual_units", "forecast_units", "model_name"}
    assert required.issubset(frame.columns)
    assert len(frame) > 0


def test_export_forecasts_writes_expected_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """export-forecasts should preserve core forecast schema in exported CSV."""
    from retail_forecasting import predict

    paths = _build_temp_paths(tmp_path)
    _write_feature_split_artifacts(paths)
    _write_best_model_artifact(paths)

    monkeypatch.setattr(predict, "build_project_paths", lambda: paths)

    generated = forecast_next_pipeline(ForecastNextConfig(use_llm_features=False))
    exported = export_forecasts_pipeline(
        ForecastExportConfig(
            input_path=generated["forecasts_latest_csv"],
            output_path=paths.artifacts_dir / "forecasts_downstream.csv",
        )
    )

    export_path = exported["forecasts_export_csv"]
    assert export_path.exists()

    frame = pd.read_csv(export_path)
    required = {"date", "store_id", "product_id", "forecast_units"}
    assert required.issubset(frame.columns)


def test_forecast_next_rejects_missing_best_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """forecast-next should fail clearly when best-model registry is unavailable."""
    from retail_forecasting import predict

    paths = _build_temp_paths(tmp_path)
    _write_feature_split_artifacts(paths)
    monkeypatch.setattr(predict, "build_project_paths", lambda: paths)

    with pytest.raises(FileNotFoundError, match="Best model registry is missing"):
        forecast_next_pipeline(ForecastNextConfig(use_llm_features=False))
