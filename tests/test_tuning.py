"""Tests for Optuna forecast model tuning pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retail_forecasting.paths import ProjectPaths
from retail_forecasting.tuning import ForecastTuningConfig, tune_forecast_models_pipeline


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
    dates = pd.date_range("2024-01-01", periods=52, freq="D")
    rows: list[dict[str, object]] = []

    for store_id in ["S1", "S2"]:
        for product_id in ["P1", "P2"]:
            for day_idx, date_value in enumerate(dates, start=1):
                units = float(18 + day_idx + (1 if store_id == "S2" else 0))
                rows.append(
                    {
                        "date": date_value,
                        "store_id": store_id,
                        "product_id": product_id,
                        "units_sold": units,
                        "price": float(6.5 + (day_idx % 5) * 0.2),
                        "discount": 0.1 if day_idx % 6 == 0 else 0.0,
                        "units_sold_lag_1": np.nan if day_idx == 1 else units - 1.0,
                        "units_sold_roll_mean_7": float(max(units - 2.0, 0.0)),
                    }
                )

    full_frame = pd.DataFrame(rows).sort_values(
        ["store_id", "product_id", "date"],
        kind="mergesort",
    )

    validation_start = pd.Timestamp("2024-02-05")
    test_start = pd.Timestamp("2024-02-14")

    train = full_frame.loc[full_frame["date"] < validation_start].reset_index(drop=True)
    validation = full_frame.loc[
        (full_frame["date"] >= validation_start) & (full_frame["date"] < test_start)
    ].reset_index(drop=True)
    test = full_frame.loc[full_frame["date"] >= test_start].reset_index(drop=True)

    full_frame.to_parquet(paths.data_processed_dir / "features_manual.parquet", index=False)
    train.to_parquet(paths.data_processed_dir / "features_train.parquet", index=False)
    validation.to_parquet(paths.data_processed_dir / "features_validation.parquet", index=False)
    test.to_parquet(paths.data_processed_dir / "features_test.parquet", index=False)


def _require_lightgbm_or_skip() -> None:
    from retail_forecasting.forecasting_models import build_forecast_model

    try:
        build_forecast_model("lightgbm", random_state=42, params={"n_estimators": 5})
    except ImportError:
        pytest.skip("LightGBM is not available in this environment")


def test_tuning_writes_optuna_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tuning should write best params JSON and summary CSV for successful studies."""
    from retail_forecasting import tuning

    _require_lightgbm_or_skip()

    paths = _build_temp_paths(tmp_path)
    _write_feature_split_artifacts(paths)
    monkeypatch.setattr(tuning, "build_project_paths", lambda: paths)

    outputs = tune_forecast_models_pipeline(
        ForecastTuningConfig(
            model="lightgbm",
            optimize_metric="wmape",
            n_trials=2,
            random_state=42,
            use_llm_features=False,
        )
    )

    assert outputs["optuna_lightgbm_best_params_json"].exists()
    assert outputs["optuna_study_summary_csv"].exists()

    summary = pd.read_csv(outputs["optuna_study_summary_csv"])
    assert "lightgbm" in summary["model_name"].astype("string").tolist()


def test_tuning_continues_when_one_model_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If one model fails during tuning, other models should still complete and write outputs."""
    from retail_forecasting import tuning

    _require_lightgbm_or_skip()

    paths = _build_temp_paths(tmp_path)
    _write_feature_split_artifacts(paths)

    monkeypatch.setattr(tuning, "build_project_paths", lambda: paths)

    original_builder = tuning.build_forecast_model

    def flaky_builder(model_name: str, *args, **kwargs):
        if model_name == "xgboost":
            raise RuntimeError("intentional xgboost failure")
        return original_builder(model_name, *args, **kwargs)

    monkeypatch.setattr(tuning, "build_forecast_model", flaky_builder)

    outputs = tune_forecast_models_pipeline(
        ForecastTuningConfig(
            model="all",
            optimize_metric="wmape",
            n_trials=2,
            random_state=42,
            use_llm_features=False,
        )
    )

    summary = pd.read_csv(outputs["optuna_study_summary_csv"])
    status_by_model = {
        str(row.model_name): str(row.status)
        for row in summary.itertuples(index=False)
    }

    assert status_by_model.get("lightgbm") == "success"
    assert status_by_model.get("xgboost") == "failed"
