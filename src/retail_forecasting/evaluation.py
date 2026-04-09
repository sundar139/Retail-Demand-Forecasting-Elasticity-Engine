"""Phase 6 time-aware forecasting evaluation and reporting utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import logging
from typing import cast

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from retail_forecasting.baselines import run_baseline_suite
from retail_forecasting.forecasting_models import (
    MODEL_ALL,
    SEGMENT_MODE_GLOBAL,
    ForecastingDataBundle,
    TrainedForecastModelArtifact,
    load_forecasting_data_bundle,
    load_model_training_registry,
    load_trained_model_artifacts,
    resolve_model_list,
    validate_segment_mode,
)
from retail_forecasting.llm_metadata import derive_llm_usage_facts, llm_usage_facts_to_dict
from retail_forecasting.paths import ProjectPaths, build_project_paths
from retail_forecasting.preprocessing import write_json
from retail_forecasting.schemas import DATE_COLUMN, PRODUCT_COLUMN, STORE_COLUMN, UNITS_COLUMN

LOGGER = logging.getLogger(__name__)

METRIC_MAPE = "mape"
METRIC_WMAPE = "wmape"
METRIC_MAE = "mae"
METRIC_RMSE = "rmse"

SUPPORTED_METRICS: frozenset[str] = frozenset({METRIC_MAPE, METRIC_WMAPE, METRIC_MAE, METRIC_RMSE})
DEFAULT_OPTIMIZE_METRIC = METRIC_WMAPE


@dataclass(frozen=True, slots=True)
class BaselineBenchmarkConfig:
    """Configuration for baseline-only benchmark runs."""

    target_column: str = UNITS_COLUMN
    use_llm_features: bool = True
    input_path: Path | None = None
    llm_features_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ForecastEvaluationConfig:
    """Configuration for evaluating baselines and trained forecast models."""

    model: str = MODEL_ALL
    optimize_metric: str = DEFAULT_OPTIMIZE_METRIC
    target_column: str = UNITS_COLUMN
    segment_mode: str = SEGMENT_MODE_GLOBAL
    use_llm_features: bool = True
    input_path: Path | None = None
    llm_features_path: Path | None = None


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Simple metric holder for consistent conversions and sorting."""

    mape: float
    wmape: float
    mae: float
    rmse: float


def compute_forecast_metrics(
    actual: Sequence[float] | pd.Series | np.ndarray,
    prediction: Sequence[float] | pd.Series | np.ndarray,
    epsilon: float = 1e-8,
) -> MetricSummary:
    """Compute robust forecast metrics with zero-demand safety."""
    actual_values = np.asarray(actual, dtype="float64")
    prediction_values = np.asarray(prediction, dtype="float64")

    if actual_values.shape != prediction_values.shape:
        raise ValueError("actual and prediction must have identical shapes")

    if actual_values.size == 0:
        return MetricSummary(mape=float("nan"), wmape=float("nan"), mae=float("nan"), rmse=float("nan"))

    errors = prediction_values - actual_values
    abs_errors = np.abs(errors)

    non_zero_mask = np.abs(actual_values) > epsilon
    if bool(non_zero_mask.any()):
        mape = float(np.mean(abs_errors[non_zero_mask] / np.abs(actual_values[non_zero_mask])) * 100.0)
    else:
        mape = float("nan")

    denominator = float(np.sum(np.abs(actual_values)))
    if denominator > epsilon:
        wmape = float(np.sum(abs_errors) / denominator * 100.0)
    else:
        wmape = float(np.mean(abs_errors) * 100.0)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    return MetricSummary(mape=mape, wmape=wmape, mae=mae, rmse=rmse)


def compute_metric_value(
    actual: Sequence[float] | pd.Series | np.ndarray,
    prediction: Sequence[float] | pd.Series | np.ndarray,
    metric_name: str,
) -> float:
    """Return a single metric value by metric name for optimization workflows."""
    metrics = compute_forecast_metrics(actual, prediction)
    normalized = _normalize_metric_name(metric_name)
    if normalized == METRIC_MAPE:
        return metrics.mape
    if normalized == METRIC_WMAPE:
        return metrics.wmape
    if normalized == METRIC_MAE:
        return metrics.mae
    return metrics.rmse


def run_baseline_benchmark_pipeline(
    config: BaselineBenchmarkConfig | None = None,
) -> dict[str, Path]:
    """Run baseline models only and persist baseline prediction + metric artifacts."""
    run_config = config if config is not None else BaselineBenchmarkConfig()

    paths = build_project_paths()
    bundle = load_forecasting_data_bundle(
        target_column=run_config.target_column,
        use_llm_features=run_config.use_llm_features,
        input_path=run_config.input_path,
        llm_features_path=run_config.llm_features_path,
        project_paths=paths,
    )

    baseline_bundle = run_baseline_suite(
        train_frame=bundle.train_frame,
        validation_frame=bundle.validation_frame,
        test_frame=bundle.test_frame,
        target_column=run_config.target_column,
    )

    validation_predictions = baseline_bundle.validation_predictions.copy(deep=True)
    test_predictions = baseline_bundle.test_predictions.copy(deep=True)
    all_predictions = pd.concat([validation_predictions, test_predictions], axis=0, ignore_index=True)

    validation_metrics = _compute_metrics_table(all_predictions, split_name="validation")
    test_metrics = _compute_metrics_table(all_predictions, split_name="test")
    segment_metrics = _compute_segment_metrics_table(all_predictions)

    outputs: dict[str, Path] = {}

    validation_prediction_path = paths.artifacts_dir / "baseline_predictions_validation.csv"
    validation_predictions.to_csv(validation_prediction_path, index=False)
    outputs["baseline_predictions_validation_csv"] = validation_prediction_path

    test_prediction_path = paths.artifacts_dir / "baseline_predictions_test.csv"
    test_predictions.to_csv(test_prediction_path, index=False)
    outputs["baseline_predictions_test_csv"] = test_prediction_path

    validation_metrics_path = paths.artifacts_dir / "baseline_metrics_validation.csv"
    validation_metrics.to_csv(validation_metrics_path, index=False)
    outputs["baseline_metrics_validation_csv"] = validation_metrics_path

    test_metrics_path = paths.artifacts_dir / "baseline_metrics_test.csv"
    test_metrics.to_csv(test_metrics_path, index=False)
    outputs["baseline_metrics_test_csv"] = test_metrics_path

    segment_metrics_path = paths.artifacts_dir / "baseline_segment_metrics.csv"
    segment_metrics.to_csv(segment_metrics_path, index=False)
    outputs["baseline_segment_metrics_csv"] = segment_metrics_path

    summary_payload: dict[str, object] = {
        "run_at_utc": _utc_now_iso(),
        "target_column": run_config.target_column,
        "training_source": bundle.training_source,
        "llm_flag_requested": bool(run_config.use_llm_features),
        "llm_columns_merged": bundle.llm_added_columns,
        "notes": bundle.notes,
        "output_paths": {name: str(path) for name, path in outputs.items()},
    }

    summary_path = write_json(summary_payload, paths.artifacts_dir / "baseline_run_summary.json")
    outputs["baseline_run_summary_json"] = summary_path
    return outputs


def evaluate_forecast_models_pipeline(
    config: ForecastEvaluationConfig | None = None,
) -> dict[str, Path]:
    """Evaluate baselines and trained models with chronological validation and test scoring."""
    run_config = config if config is not None else ForecastEvaluationConfig()
    optimize_metric = _normalize_metric_name(run_config.optimize_metric)
    requested_models = resolve_model_list(run_config.model)
    segment_mode = validate_segment_mode(run_config.segment_mode)

    paths = build_project_paths()
    bundle = load_forecasting_data_bundle(
        target_column=run_config.target_column,
        use_llm_features=run_config.use_llm_features,
        input_path=run_config.input_path,
        llm_features_path=run_config.llm_features_path,
        project_paths=paths,
    )

    baseline_bundle = run_baseline_suite(
        train_frame=bundle.train_frame,
        validation_frame=bundle.validation_frame,
        test_frame=bundle.test_frame,
        target_column=run_config.target_column,
    )

    prediction_frames: list[pd.DataFrame] = [
        baseline_bundle.validation_predictions,
        baseline_bundle.test_predictions,
    ]

    warnings: list[str] = []

    trained_artifacts = load_trained_model_artifacts(
        model=run_config.model,
        segment_mode=segment_mode,
        project_paths=paths,
    )

    if not trained_artifacts:
        warnings.append("No trained model artifacts found; evaluation includes baselines only")

    for model_name in requested_models:
        artifact = trained_artifacts.get(model_name)
        if artifact is None:
            warnings.append(f"Model {model_name} is not available in trained artifacts and was skipped")
            continue

        try:
            validation_predictions = artifact.predict(bundle.validation_frame)
            prediction_frames.append(
                _build_prediction_frame(
                    source_frame=bundle.validation_frame,
                    predictions=validation_predictions,
                    model_name=model_name,
                    model_family=artifact.model_family,
                    split_name="validation",
                    target_column=run_config.target_column,
                )
            )
        except Exception as exc:  # noqa: BLE001
            warning = f"Validation prediction failed for {model_name}: {exc}"
            warnings.append(warning)
            LOGGER.exception(warning)
            continue

        try:
            test_predictions = artifact.predict(bundle.test_frame)
            prediction_frames.append(
                _build_prediction_frame(
                    source_frame=bundle.test_frame,
                    predictions=test_predictions,
                    model_name=model_name,
                    model_family=artifact.model_family,
                    split_name="test",
                    target_column=run_config.target_column,
                )
            )
        except Exception as exc:  # noqa: BLE001
            warning = f"Test prediction failed for {model_name}: {exc}"
            warnings.append(warning)
            LOGGER.exception(warning)

    all_predictions = pd.concat(prediction_frames, axis=0, ignore_index=True)

    validation_predictions = all_predictions.loc[all_predictions["split"] == "validation"].copy()
    test_predictions = all_predictions.loc[all_predictions["split"] == "test"].copy()

    validation_metrics = _compute_metrics_table(all_predictions, split_name="validation")
    test_metrics = _compute_metrics_table(all_predictions, split_name="test")
    segment_metrics = _compute_segment_metrics_table(all_predictions)

    if validation_metrics.empty:
        raise ValueError("Validation metrics are empty; no models produced usable predictions")

    best_row = select_best_model(validation_metrics, optimize_metric=optimize_metric)
    model_metadata = _build_model_metadata_map(
        bundle=bundle,
        trained_artifacts=trained_artifacts,
        llm_requested=run_config.use_llm_features,
        project_paths=paths,
    )

    best_registry = _build_best_model_registry(
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        segment_metrics=segment_metrics,
        optimize_metric=optimize_metric,
        metadata_map=model_metadata,
    )

    outputs: dict[str, Path] = {}

    validation_prediction_path = paths.artifacts_dir / "forecast_predictions_validation.csv"
    validation_predictions.to_csv(validation_prediction_path, index=False)
    outputs["forecast_predictions_validation_csv"] = validation_prediction_path

    test_prediction_path = paths.artifacts_dir / "forecast_predictions_test.csv"
    test_predictions.to_csv(test_prediction_path, index=False)
    outputs["forecast_predictions_test_csv"] = test_prediction_path

    validation_metrics_path = paths.artifacts_dir / "forecast_metrics_validation.csv"
    validation_metrics.to_csv(validation_metrics_path, index=False)
    outputs["forecast_metrics_validation_csv"] = validation_metrics_path

    test_metrics_path = paths.artifacts_dir / "forecast_metrics_test.csv"
    test_metrics.to_csv(test_metrics_path, index=False)
    outputs["forecast_metrics_test_csv"] = test_metrics_path

    segment_metrics_path = paths.artifacts_dir / "forecast_segment_metrics.csv"
    segment_metrics.to_csv(segment_metrics_path, index=False)
    outputs["forecast_segment_metrics_csv"] = segment_metrics_path

    best_registry_path = paths.artifacts_dir / "best_model_registry.csv"
    best_registry.to_csv(best_registry_path, index=False)
    outputs["best_model_registry_csv"] = best_registry_path

    reports_outputs = _generate_evaluation_report_artifacts(
        all_predictions=all_predictions,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        segment_metrics=segment_metrics,
        optimize_metric=optimize_metric,
        best_model_name=str(best_row["model_name"]),
        project_paths=paths,
    )
    outputs.update(reports_outputs)

    best_model_metadata = model_metadata.get(str(best_row["model_name"]), {})
    best_model_llm_columns = str(best_model_metadata.get("llm_feature_columns", ""))
    best_model_llm_usage = derive_llm_usage_facts(
        llm_requested=run_config.use_llm_features,
        llm_feature_columns_used=best_model_llm_columns,
        project_paths=paths,
    )

    summary_payload: dict[str, object] = {
        "run_at_utc": _utc_now_iso(),
        "optimize_metric": optimize_metric,
        "best_model_name": str(best_row["model_name"]),
        "best_model_family": str(best_row["model_family"]),
        "best_validation_metrics": {
            METRIC_MAPE: float(best_row[METRIC_MAPE]),
            METRIC_WMAPE: float(best_row[METRIC_WMAPE]),
            METRIC_MAE: float(best_row[METRIC_MAE]),
            METRIC_RMSE: float(best_row[METRIC_RMSE]),
        },
        "training_source": bundle.training_source,
        "llm_requested": bool(run_config.use_llm_features),
        "llm_flag_requested": bool(run_config.use_llm_features),
        "llm_columns_merged": bundle.llm_added_columns,
        "ollama_reachable": best_model_llm_usage.ollama_reachable,
        "planner_model_available": best_model_llm_usage.planner_model_available,
        "llm_feature_file_exists": best_model_llm_usage.llm_feature_file_exists,
        "llm_output_feature_count": int(best_model_llm_usage.llm_output_feature_count),
        "llm_features_actually_used": bool(best_model_llm_usage.llm_features_actually_used),
        "llm_feature_columns_used": (
            best_model_llm_usage.llm_feature_columns_used
            if best_model_llm_usage.llm_features_actually_used
            else []
        ),
        "llm_features_actually_used_in_best_model": bool(
            best_model_llm_usage.llm_features_actually_used
        ),
        "llm_usage_facts": llm_usage_facts_to_dict(best_model_llm_usage),
        "warnings": warnings,
        "output_paths": {name: str(path) for name, path in outputs.items()},
    }

    summary_path = write_json(summary_payload, paths.artifacts_dir / "forecast_evaluation_summary.json")
    outputs["forecast_evaluation_summary_json"] = summary_path
    return outputs


def select_best_model(validation_metrics: pd.DataFrame, optimize_metric: str = DEFAULT_OPTIMIZE_METRIC) -> pd.Series:
    """Select best model row based on configured optimization metric."""
    metric_name = _normalize_metric_name(optimize_metric)
    if validation_metrics.empty:
        raise ValueError("validation_metrics is empty")
    if metric_name not in validation_metrics.columns:
        raise ValueError(f"Metric column '{metric_name}' is missing in validation_metrics")

    candidates = validation_metrics.copy(deep=True)
    metric_values = pd.to_numeric(candidates[metric_name], errors="coerce").astype("float64")
    candidates = candidates.loc[metric_values.notna()].copy()
    if candidates.empty:
        raise ValueError(f"No finite values found for optimization metric '{metric_name}'")

    best_index = candidates[metric_name].astype("float64").idxmin()
    best_row = candidates.loc[best_index]
    if isinstance(best_row, pd.DataFrame):
        best_row = best_row.iloc[0]
    return cast(pd.Series, best_row)


def load_forecast_metrics(
    split: str,
    metrics_path: Path | None = None,
    project_paths: ProjectPaths | None = None,
) -> pd.DataFrame:
    """Load forecast metrics table for validation or test split."""
    normalized = split.strip().lower()
    if normalized not in {"validation", "test"}:
        raise ValueError("split must be either 'validation' or 'test'")

    paths = project_paths if project_paths is not None else build_project_paths()
    default_path = paths.artifacts_dir / f"forecast_metrics_{normalized}.csv"
    target = default_path if metrics_path is None else Path(metrics_path)
    if not target.is_absolute():
        target = paths.project_root / target

    if not target.exists():
        raise FileNotFoundError(
            f"Missing forecast metrics artifact: {target}. Run 'evaluate-forecast-models' first."
        )

    return pd.read_csv(target)


def load_best_model_registry(
    registry_path: Path | None = None,
    project_paths: ProjectPaths | None = None,
) -> pd.DataFrame:
    """Load best-model registry artifact."""
    paths = project_paths if project_paths is not None else build_project_paths()
    target = paths.artifacts_dir / "best_model_registry.csv" if registry_path is None else Path(registry_path)
    if not target.is_absolute():
        target = paths.project_root / target

    if not target.exists():
        raise FileNotFoundError(
            f"Missing best-model registry artifact: {target}. Run 'evaluate-forecast-models' first."
        )

    return pd.read_csv(target)


def rolling_backtest_cutoffs(
    frame: pd.DataFrame,
    initial_train_days: int,
    horizon_days: int,
    step_days: int,
    date_column: str = DATE_COLUMN,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Optional expanding-window cutoff generator for backtest workflows."""
    if initial_train_days <= 0 or horizon_days <= 0 or step_days <= 0:
        raise ValueError("initial_train_days, horizon_days, and step_days must be positive")

    unique_dates = sorted(pd.to_datetime(frame[date_column], errors="coerce").dropna().dt.floor("D").unique())
    if len(unique_dates) < initial_train_days + horizon_days:
        return []

    cutoffs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    train_end_index = initial_train_days - 1

    while train_end_index + horizon_days < len(unique_dates):
        validation_start = pd.Timestamp(unique_dates[train_end_index + 1])
        validation_end = pd.Timestamp(unique_dates[train_end_index + horizon_days])
        cutoffs.append((validation_start, validation_end))
        train_end_index += step_days

    return cutoffs


def _normalize_metric_name(metric_name: str) -> str:
    normalized = metric_name.strip().lower()
    if normalized not in SUPPORTED_METRICS:
        allowed = ", ".join(sorted(SUPPORTED_METRICS))
        raise ValueError(f"Unsupported metric '{metric_name}'. Allowed values: {allowed}")
    return normalized


def _build_prediction_frame(
    source_frame: pd.DataFrame,
    predictions: Sequence[float] | np.ndarray,
    model_name: str,
    model_family: str,
    split_name: str,
    target_column: str,
) -> pd.DataFrame:
    prediction_values = np.asarray(predictions, dtype="float64")
    if len(prediction_values) != len(source_frame):
        raise ValueError(
            f"Prediction length mismatch for {model_name}/{split_name}: "
            f"expected {len(source_frame)} got {len(prediction_values)}"
        )

    output = source_frame[[DATE_COLUMN, STORE_COLUMN, PRODUCT_COLUMN, target_column]].copy(deep=True)
    output = output.rename(columns={target_column: "actual"})
    output["prediction"] = prediction_values
    output["prediction_error"] = output["prediction"] - output["actual"]
    output["prediction_abs_error"] = output["prediction_error"].abs()
    output["model_name"] = model_name
    output["model_family"] = model_family
    output["split"] = split_name
    return output


def _compute_metrics_table(predictions: pd.DataFrame, split_name: str) -> pd.DataFrame:
    split_predictions = predictions.loc[predictions["split"] == split_name].copy()
    if split_predictions.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "model_name",
                "model_family",
                "n_obs",
                METRIC_MAPE,
                METRIC_WMAPE,
                METRIC_MAE,
                METRIC_RMSE,
            ]
        )

    rows: list[dict[str, object]] = []
    grouped = split_predictions.groupby(["model_name", "model_family"], sort=True)

    for (model_name, model_family), group in grouped:
        metrics = compute_forecast_metrics(group["actual"], group["prediction"])
        rows.append(
            {
                "split": split_name,
                "model_name": str(model_name),
                "model_family": str(model_family),
                "n_obs": int(len(group)),
                METRIC_MAPE: float(metrics.mape),
                METRIC_WMAPE: float(metrics.wmape),
                METRIC_MAE: float(metrics.mae),
                METRIC_RMSE: float(metrics.rmse),
            }
        )

    return pd.DataFrame.from_records(rows).sort_values(by=[METRIC_WMAPE], kind="mergesort").reset_index(
        drop=True
    )


def _compute_segment_metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "model_name",
                "model_family",
                "segment_key",
                "n_obs",
                METRIC_MAPE,
                METRIC_WMAPE,
                METRIC_MAE,
                METRIC_RMSE,
            ]
        )

    working = predictions.copy(deep=True)
    working["segment_key"] = "product_id=" + working[PRODUCT_COLUMN].astype("string")

    rows: list[dict[str, object]] = []
    grouped = working.groupby(["split", "model_name", "model_family", "segment_key"], sort=True)

    for (split_name, model_name, model_family, segment_key), group in grouped:
        metrics = compute_forecast_metrics(group["actual"], group["prediction"])
        rows.append(
            {
                "split": str(split_name),
                "model_name": str(model_name),
                "model_family": str(model_family),
                "segment_key": str(segment_key),
                "n_obs": int(len(group)),
                METRIC_MAPE: float(metrics.mape),
                METRIC_WMAPE: float(metrics.wmape),
                METRIC_MAE: float(metrics.mae),
                METRIC_RMSE: float(metrics.rmse),
            }
        )

    return pd.DataFrame.from_records(rows).sort_values(
        by=["split", "segment_key", METRIC_WMAPE],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_model_metadata_map(
    bundle: ForecastingDataBundle,
    trained_artifacts: Mapping[str, TrainedForecastModelArtifact],
    llm_requested: bool,
    project_paths: ProjectPaths,
) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}

    # Baselines do not consume feature matrices and therefore never use optional LLM features directly.
    baseline_names = {
        "naive_last",
        "seasonal_naive_7",
        "rolling_mean_7",
        "rolling_mean_28",
        "median_expanding",
    }

    for baseline_name in baseline_names:
        metadata[baseline_name] = {
            "model_family": "baseline",
            "training_source": bundle.training_source,
            "feature_count": 0,
            "llm_features_used": False,
            "llm_feature_columns": "",
            "llm_requested": bool(llm_requested),
            "model_artifact_path": "",
        }

    training_registry_map: dict[str, str] = {}
    try:
        registry = load_model_training_registry(project_paths=project_paths)
        successful = registry.loc[registry["status"].astype("string") == "success"].copy()
        successful = successful.sort_values(by=["trained_at_utc"], kind="mergesort")
        successful = successful.drop_duplicates(subset=["model_name"], keep="last")
        training_registry_map = {
            str(row.model_name): str(row.model_path)
            for row in successful.itertuples(index=False)
            if str(getattr(row, "model_path", ""))
        }
    except FileNotFoundError:
        training_registry_map = {}

    for model_name, artifact in trained_artifacts.items():
        llm_columns_used = [
            column_name
            for column_name in artifact.llm_feature_columns_in_matrix
            if column_name in artifact.feature_columns
        ]
        model_llm_usage = derive_llm_usage_facts(
            llm_requested=llm_requested,
            llm_feature_columns_used=llm_columns_used,
            project_paths=project_paths,
        )
        llm_columns_text = (
            "|".join(model_llm_usage.llm_feature_columns_used)
            if model_llm_usage.llm_features_actually_used
            else ""
        )
        metadata[model_name] = {
            "model_family": artifact.model_family,
            "training_source": artifact.training_source,
            "feature_count": int(len(artifact.feature_columns)),
            "llm_features_used": bool(model_llm_usage.llm_features_actually_used),
            "llm_feature_columns": llm_columns_text,
            "llm_requested": bool(llm_requested),
            "model_artifact_path": training_registry_map.get(model_name, ""),
        }

    return metadata


def _build_best_model_registry(
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    segment_metrics: pd.DataFrame,
    optimize_metric: str,
    metadata_map: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    best_overall = select_best_model(validation_metrics, optimize_metric=optimize_metric)
    best_model_name = str(best_overall["model_name"])

    test_match = test_metrics.loc[test_metrics["model_name"] == best_model_name].copy()
    test_row = test_match.iloc[0] if not test_match.empty else None

    overall_metadata = dict(metadata_map.get(best_model_name, {}))

    rows: list[dict[str, object]] = [
        {
            "selected_at_utc": _utc_now_iso(),
            "scope": "overall",
            "segment_key": "all",
            "optimize_metric": optimize_metric,
            "model_name": best_model_name,
            "model_family": str(best_overall.get("model_family", "unknown")),
            "validation_mape": _coerce_float(best_overall.get(METRIC_MAPE)),
            "validation_wmape": _coerce_float(best_overall.get(METRIC_WMAPE)),
            "validation_mae": _coerce_float(best_overall.get(METRIC_MAE)),
            "validation_rmse": _coerce_float(best_overall.get(METRIC_RMSE)),
            "test_mape": _coerce_float(test_row.get(METRIC_MAPE)) if test_row is not None else float("nan"),
            "test_wmape": _coerce_float(test_row.get(METRIC_WMAPE)) if test_row is not None else float("nan"),
            "test_mae": _coerce_float(test_row.get(METRIC_MAE)) if test_row is not None else float("nan"),
            "test_rmse": _coerce_float(test_row.get(METRIC_RMSE)) if test_row is not None else float("nan"),
            "training_source": str(overall_metadata.get("training_source", "unknown")),
            "feature_count": _coerce_int(overall_metadata.get("feature_count", 0)),
            "llm_features_used": bool(overall_metadata.get("llm_features_used", False)),
            "llm_feature_columns": str(overall_metadata.get("llm_feature_columns", "")),
            "model_artifact_path": str(overall_metadata.get("model_artifact_path", "")),
        }
    ]

    validation_segment = segment_metrics.loc[segment_metrics["split"] == "validation"].copy()
    if not validation_segment.empty:
        validation_segment = validation_segment.loc[
            pd.to_numeric(validation_segment[optimize_metric], errors="coerce").notna()
        ].copy()

    if not validation_segment.empty:
        for segment_key, group in validation_segment.groupby("segment_key", sort=True):
            best_index = group[optimize_metric].astype("float64").idxmin()
            best_segment_row_obj = group.loc[best_index]
            best_segment_row = (
                best_segment_row_obj.iloc[0]
                if isinstance(best_segment_row_obj, pd.DataFrame)
                else best_segment_row_obj
            )
            segment_model_name = str(best_segment_row["model_name"])
            segment_metadata = dict(metadata_map.get(segment_model_name, {}))

            rows.append(
                {
                    "selected_at_utc": _utc_now_iso(),
                    "scope": "per-product",
                    "segment_key": str(segment_key),
                    "optimize_metric": optimize_metric,
                    "model_name": segment_model_name,
                    "model_family": str(best_segment_row.get("model_family", "unknown")),
                    "validation_mape": _coerce_float(best_segment_row.get(METRIC_MAPE)),
                    "validation_wmape": _coerce_float(best_segment_row.get(METRIC_WMAPE)),
                    "validation_mae": _coerce_float(best_segment_row.get(METRIC_MAE)),
                    "validation_rmse": _coerce_float(best_segment_row.get(METRIC_RMSE)),
                    "test_mape": float("nan"),
                    "test_wmape": float("nan"),
                    "test_mae": float("nan"),
                    "test_rmse": float("nan"),
                    "training_source": str(segment_metadata.get("training_source", "unknown")),
                    "feature_count": _coerce_int(segment_metadata.get("feature_count", 0)),
                    "llm_features_used": bool(segment_metadata.get("llm_features_used", False)),
                    "llm_feature_columns": str(segment_metadata.get("llm_feature_columns", "")),
                    "model_artifact_path": str(segment_metadata.get("model_artifact_path", "")),
                }
            )

    return pd.DataFrame.from_records(rows)


def _generate_evaluation_report_artifacts(
    all_predictions: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    segment_metrics: pd.DataFrame,
    optimize_metric: str,
    best_model_name: str,
    project_paths: ProjectPaths,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    reports_dir = project_paths.reports_figures_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison_plot_path = reports_dir / "forecast_model_comparison.png"
    _plot_model_comparison(validation_metrics, test_metrics, comparison_plot_path)
    outputs["forecast_model_comparison_png"] = comparison_plot_path

    histogram_path = reports_dir / "forecast_error_distribution.png"
    _plot_error_histogram(
        all_predictions=all_predictions,
        model_name=best_model_name,
        split_name="validation",
        output_path=histogram_path,
    )
    outputs["forecast_error_distribution_png"] = histogram_path

    prediction_plot_path = reports_dir / "forecast_predicted_vs_actual_sample.png"
    _plot_predicted_vs_actual_sample(
        all_predictions=all_predictions,
        model_name=best_model_name,
        split_name="validation",
        output_path=prediction_plot_path,
    )
    outputs["forecast_predicted_vs_actual_png"] = prediction_plot_path

    segment_extremes_path = reports_dir / "forecast_best_worst_segments.csv"
    _write_segment_extremes(
        segment_metrics=segment_metrics,
        model_name=best_model_name,
        optimize_metric=optimize_metric,
        output_path=segment_extremes_path,
    )
    outputs["forecast_best_worst_segments_csv"] = segment_extremes_path

    return outputs


def _plot_model_comparison(
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(10, 6))
    ax = figure.add_subplot(1, 1, 1)

    model_names = sorted(
        set(validation_metrics["model_name"].astype("string").tolist())
        | set(test_metrics["model_name"].astype("string").tolist())
    )

    if not model_names:
        ax.set_title("Model Comparison (no data)")
        figure.tight_layout()
        figure.savefig(output_path, dpi=120)
        plt.close(figure)
        return

    x_positions = np.arange(len(model_names))
    width = 0.35

    validation_lookup = {
        str(row.model_name): _coerce_float(getattr(row, "wmape", np.nan), default=float("nan"))
        for row in validation_metrics.itertuples(index=False)
    }
    test_lookup = {
        str(row.model_name): _coerce_float(getattr(row, "wmape", np.nan), default=float("nan"))
        for row in test_metrics.itertuples(index=False)
    }

    validation_values = [validation_lookup.get(name, np.nan) for name in model_names]
    test_values = [test_lookup.get(name, np.nan) for name in model_names]

    ax.bar(x_positions - width / 2.0, validation_values, width=width, label="validation")
    ax.bar(x_positions + width / 2.0, test_values, width=width, label="test")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_names, rotation=35, ha="right")
    ax.set_ylabel("wMAPE (%)")
    ax.set_title("Forecast Model Comparison")
    ax.legend(loc="best")

    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def _plot_error_histogram(
    all_predictions: pd.DataFrame,
    model_name: str,
    split_name: str,
    output_path: Path,
) -> None:
    subset = all_predictions.loc[
        (all_predictions["model_name"] == model_name) & (all_predictions["split"] == split_name)
    ].copy()

    figure = plt.figure(figsize=(9, 5))
    ax = figure.add_subplot(1, 1, 1)

    if subset.empty:
        ax.set_title("Absolute Error Distribution (no data)")
    else:
        absolute_errors = pd.to_numeric(subset["prediction_abs_error"], errors="coerce").dropna()
        ax.hist(absolute_errors.to_numpy(dtype="float64"), bins=30, edgecolor="black", alpha=0.8)
        ax.set_title(f"Absolute Error Distribution: {model_name} ({split_name})")
        ax.set_xlabel("Absolute error")
        ax.set_ylabel("Count")

    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def _plot_predicted_vs_actual_sample(
    all_predictions: pd.DataFrame,
    model_name: str,
    split_name: str,
    output_path: Path,
) -> None:
    subset = all_predictions.loc[
        (all_predictions["model_name"] == model_name) & (all_predictions["split"] == split_name)
    ].copy()

    figure = plt.figure(figsize=(10, 5))
    ax = figure.add_subplot(1, 1, 1)

    if subset.empty:
        ax.set_title("Predicted vs Actual (no data)")
        figure.tight_layout()
        figure.savefig(output_path, dpi=120)
        plt.close(figure)
        return

    segment_counts = subset[PRODUCT_COLUMN].astype("string").value_counts()
    sample_product = str(segment_counts.index[0])

    sample = subset.loc[subset[PRODUCT_COLUMN].astype("string") == sample_product].copy()
    sample = sample.sort_values(by=[DATE_COLUMN], kind="mergesort")

    ax.plot(sample[DATE_COLUMN], sample["actual"], label="actual", linewidth=2)
    ax.plot(sample[DATE_COLUMN], sample["prediction"], label="prediction", linewidth=2)
    ax.set_title(f"Predicted vs Actual ({model_name}, product={sample_product})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")
    ax.legend(loc="best")

    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def _write_segment_extremes(
    segment_metrics: pd.DataFrame,
    model_name: str,
    optimize_metric: str,
    output_path: Path,
) -> None:
    subset = segment_metrics.loc[
        (segment_metrics["split"] == "validation") & (segment_metrics["model_name"] == model_name)
    ].copy()

    if subset.empty:
        pd.DataFrame(
            columns=["rank_group", "segment_key", METRIC_MAPE, METRIC_WMAPE, METRIC_MAE, METRIC_RMSE]
        ).to_csv(output_path, index=False)
        return

    subset = subset.sort_values(by=[optimize_metric], kind="mergesort").reset_index(drop=True)

    best = subset.head(10).copy()
    best["rank_group"] = "best"

    worst = subset.tail(10).copy()
    worst["rank_group"] = "worst"

    output = pd.concat([best, worst], axis=0, ignore_index=True)
    output.to_csv(output_path, index=False)


def _coerce_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _coerce_float(value: object, default: float = float("nan")) -> float:
    coerced = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(coerced):
        return default
    try:
        return float(coerced)
    except (TypeError, ValueError):
        return default


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()
