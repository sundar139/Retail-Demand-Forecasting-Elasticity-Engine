"""Tests for forecasting evaluation logic and resilience."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from retail_forecasting.evaluation import (
    ForecastEvaluationConfig,
    compute_forecast_metrics,
    evaluate_forecast_models_pipeline,
    select_best_model,
)
from retail_forecasting.forecasting_models import validate_chronological_split_integrity
from retail_forecasting.paths import ProjectPaths


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
    dates = pd.date_range("2024-01-01", periods=56, freq="D")
    rows: list[dict[str, object]] = []

    for store_id in ["S1", "S2"]:
        for product_id in ["P1", "P2"]:
            for day_idx, date_value in enumerate(dates, start=1):
                units = float(15 + day_idx + (2 if product_id == "P2" else 0))
                rows.append(
                    {
                        "date": date_value,
                        "store_id": store_id,
                        "product_id": product_id,
                        "units_sold": units,
                        "price": float(7.0 + (day_idx % 6) * 0.2),
                        "discount": 0.1 if day_idx % 4 == 0 else 0.0,
                        "units_sold_lag_1": np.nan if day_idx == 1 else units - 1.0,
                        "units_sold_roll_mean_7": float(max(units - 2.0, 0.0)),
                    }
                )

    full_frame = pd.DataFrame(rows).sort_values(
        ["store_id", "product_id", "date"],
        kind="mergesort",
    )

    validation_start = pd.Timestamp("2024-02-10")
    test_start = pd.Timestamp("2024-02-20")

    train = full_frame.loc[full_frame["date"] < validation_start].reset_index(drop=True)
    validation = full_frame.loc[
        (full_frame["date"] >= validation_start) & (full_frame["date"] < test_start)
    ].reset_index(drop=True)
    test = full_frame.loc[full_frame["date"] >= test_start].reset_index(drop=True)

    full_frame.to_parquet(paths.data_processed_dir / "features_manual.parquet", index=False)
    train.to_parquet(paths.data_processed_dir / "features_train.parquet", index=False)
    validation.to_parquet(paths.data_processed_dir / "features_validation.parquet", index=False)
    test.to_parquet(paths.data_processed_dir / "features_test.parquet", index=False)


class _WorkingArtifact:
    def __init__(self, llm_columns: list[str] | None = None) -> None:
        self.model_family = "lightgbm"
        self.training_source = "split_feature_artifacts"
        base_columns = ["units_sold_lag_1", "units_sold_roll_mean_7", "price", "discount"]
        self.llm_feature_columns_in_matrix = list(llm_columns or [])
        self.feature_columns = base_columns + [
            column_name for column_name in self.llm_feature_columns_in_matrix if column_name not in base_columns
        ]
        self.llm_features_used = bool(self.llm_feature_columns_in_matrix)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        actual = pd.to_numeric(frame["units_sold"], errors="coerce")
        if actual.notna().all():
            return actual.to_numpy(dtype="float64")

        lag = pd.to_numeric(frame["units_sold_lag_1"], errors="coerce").fillna(0.0)
        roll = pd.to_numeric(frame["units_sold_roll_mean_7"], errors="coerce").fillna(0.0)
        return (0.55 * lag + 0.45 * roll).to_numpy(dtype="float64")


class _FailingArtifact:
    def __init__(self) -> None:
        self.model_family = "xgboost"
        self.training_source = "split_feature_artifacts"
        self.feature_columns = ["units_sold_lag_1"]
        self.llm_features_used = False
        self.llm_feature_columns_in_matrix: list[str] = []

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raise RuntimeError("intentional prediction failure")


def test_compute_forecast_metrics_zero_demand_safety() -> None:
    """wMAPE should remain finite when zero-demand rows are present."""
    actual = np.array([0.0, 0.0, 10.0])
    prediction = np.array([1.0, 0.0, 8.0])

    metrics = compute_forecast_metrics(actual, prediction)

    assert np.isfinite(metrics.wmape)
    assert round(metrics.wmape, 6) == 30.0
    assert round(metrics.mape, 6) == 20.0


def test_validate_chronological_split_integrity_detects_overlap() -> None:
    """Chronological guardrail should reject overlapping split boundaries."""
    train = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "x": [1, 2]})
    validation = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "x": [3, 4]})
    test = pd.DataFrame({"date": pd.to_datetime(["2024-01-04"]), "x": [5]})

    with pytest.raises(ValueError, match="train split overlaps validation period"):
        validate_chronological_split_integrity(train, validation, test)


def test_select_best_model_prefers_lowest_wmape() -> None:
    """Best-model selector should choose minimum validation wMAPE."""
    validation_metrics = pd.DataFrame(
        {
            "model_name": ["naive_last", "lightgbm", "xgboost"],
            "model_family": ["baseline", "lightgbm", "xgboost"],
            "mape": [25.0, 18.0, 20.0],
            "wmape": [22.0, 15.0, 19.0],
            "mae": [5.0, 3.0, 4.0],
            "rmse": [6.0, 4.0, 5.0],
        }
    )

    best = select_best_model(validation_metrics, optimize_metric="wmape")

    assert str(best["model_name"]) == "lightgbm"


def test_evaluation_pipeline_handles_one_model_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluation should continue when one trained model fails and others still succeed."""
    from retail_forecasting import evaluation

    paths = _build_temp_paths(tmp_path)
    _write_feature_split_artifacts(paths)

    monkeypatch.setattr(evaluation, "build_project_paths", lambda: paths)
    monkeypatch.setattr(
        evaluation,
        "load_trained_model_artifacts",
        lambda *args, **kwargs: {
            "lightgbm": _WorkingArtifact(),
            "xgboost": _FailingArtifact(),
        },
    )

    outputs = evaluate_forecast_models_pipeline(
        ForecastEvaluationConfig(
            model="all",
            optimize_metric="wmape",
            use_llm_features=False,
            segment_mode="global",
        )
    )

    assert outputs["forecast_metrics_validation_csv"].exists()
    assert outputs["forecast_metrics_test_csv"].exists()
    assert outputs["best_model_registry_csv"].exists()

    validation_metrics = pd.read_csv(outputs["forecast_metrics_validation_csv"])
    assert "lightgbm" in validation_metrics["model_name"].astype("string").tolist()


def test_evaluation_summary_reports_llm_usage_when_columns_are_actually_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluation summary should report LLM usage only when file+summary+matrix usage all align."""
    from retail_forecasting import evaluation

    paths = _build_temp_paths(tmp_path)
    _write_feature_split_artifacts(paths)

    pd.DataFrame(
        [{"date": "2024-01-01", "store_id": "S1", "product_id": "P1", "llm_signal": 1.0}]
    ).to_parquet(paths.data_processed_dir / "features_llm.parquet", index=False)

    (paths.artifacts_dir / "llm_features_summary.json").write_text(
        json.dumps(
            {
                "llm_requested": True,
                "ollama_reachable": True,
                "planner_model_available": True,
                "output_feature_count": 1,
                "output_feature_names": ["llm_signal"],
                "output_paths": {
                    "features_llm_parquet": str(paths.data_processed_dir / "features_llm.parquet")
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(evaluation, "build_project_paths", lambda: paths)
    monkeypatch.setattr(
        evaluation,
        "load_trained_model_artifacts",
        lambda *args, **kwargs: {
            "lightgbm": _WorkingArtifact(llm_columns=["llm_signal"]),
        },
    )

    outputs = evaluate_forecast_models_pipeline(
        ForecastEvaluationConfig(
            model="lightgbm",
            optimize_metric="wmape",
            use_llm_features=True,
            segment_mode="global",
        )
    )

    summary_payload = json.loads(outputs["forecast_evaluation_summary_json"].read_text(encoding="utf-8"))
    assert bool(summary_payload.get("llm_features_actually_used", False)) is True
    assert summary_payload.get("llm_feature_columns_used", []) == ["llm_signal"]
