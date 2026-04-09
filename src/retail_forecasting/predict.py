"""Phase 7 prediction and forecast export workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

from retail_forecasting.forecasting_models import (
    DEFAULT_TARGET_COLUMN,
    ForecastingDataBundle,
    TrainedForecastModelArtifact,
    load_forecasting_data_bundle,
)
from retail_forecasting.paths import ProjectPaths, build_project_paths
from retail_forecasting.preprocessing import write_json
from retail_forecasting.schemas import DATE_COLUMN, PRODUCT_COLUMN, STORE_COLUMN


@dataclass(frozen=True, slots=True)
class ForecastNextConfig:
    """Configuration for generating latest forecasts from current best model artifact."""

    use_llm_features: bool = True
    output_path: Path | None = None
    input_path: Path | None = None
    llm_features_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ForecastExportConfig:
    """Configuration for exporting previously generated forecast outputs."""

    input_path: Path | None = None
    output_path: Path | None = None


def forecast_next_pipeline(config: ForecastNextConfig | None = None) -> dict[str, Path]:
    """Generate latest split forecasts using the current best registered model."""
    run_config = config if config is not None else ForecastNextConfig()
    paths = build_project_paths()

    best_model_record = _load_best_model_record(paths)
    model_name = str(best_model_record.get("model_name", ""))
    model_family = str(best_model_record.get("model_family", ""))
    model_path_text = str(best_model_record.get("model_artifact_path", "") or "")

    if not model_path_text:
        raise ValueError(
            "Best model registry does not contain a model artifact path. "
            "Train forecast models before running forecast-next."
        )

    model_path = Path(model_path_text)
    if not model_path.is_absolute():
        model_path = paths.project_root / model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Best model artifact path does not exist: {model_path}. "
            "Re-run train-forecast-models and evaluate-forecast-models."
        )

    loaded_model = joblib.load(model_path)
    if not isinstance(loaded_model, TrainedForecastModelArtifact):
        raise ValueError(
            "Best model artifact has an unexpected type and cannot be used for forecasting"
        )

    loaded_model.llm_feature_columns_in_matrix = [
        column_name
        for column_name in loaded_model.llm_feature_columns_in_matrix
        if column_name in loaded_model.feature_columns
    ]
    loaded_model.llm_features_used = bool(loaded_model.llm_feature_columns_in_matrix)

    bundle = load_forecasting_data_bundle(
        target_column=DEFAULT_TARGET_COLUMN,
        use_llm_features=run_config.use_llm_features,
        input_path=run_config.input_path,
        llm_features_path=run_config.llm_features_path,
        project_paths=paths,
    )
    source_split_name, latest_frame = _select_latest_split(bundle)

    predictions = loaded_model.predict(latest_frame)

    output_frame = latest_frame[[DATE_COLUMN, STORE_COLUMN, PRODUCT_COLUMN, DEFAULT_TARGET_COLUMN]].copy(deep=True)
    output_frame = output_frame.rename(columns={DEFAULT_TARGET_COLUMN: "actual_units"})
    output_frame["forecast_units"] = pd.Series(predictions, index=output_frame.index).astype("float64")
    output_frame["model_name"] = model_name
    output_frame["model_family"] = model_family
    output_frame["source_split"] = source_split_name
    output_frame["generated_at_utc"] = _utc_now_iso()

    output_path = _resolve_output_path(
        default_path=paths.artifacts_dir / "forecasts_latest.csv",
        requested=run_config.output_path,
        project_paths=paths,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output_path, index=False)

    summary_payload: dict[str, object] = {
        "generated_at_utc": _utc_now_iso(),
        "best_model_name": model_name,
        "best_model_family": model_family,
        "model_artifact_path": str(model_path),
        "source_split": source_split_name,
        "row_count": int(len(output_frame)),
        "llm_features_used": bool(loaded_model.llm_feature_columns_in_matrix),
        "llm_feature_columns": loaded_model.llm_feature_columns_in_matrix,
        "output_path": str(output_path),
    }
    summary_path = write_json(summary_payload, paths.artifacts_dir / "forecast_export_summary.json")

    return {
        "forecasts_latest_csv": output_path,
        "forecast_export_summary_json": summary_path,
    }


def export_forecasts_pipeline(config: ForecastExportConfig | None = None) -> dict[str, Path]:
    """Export forecast predictions to downstream CSV target and write export summary metadata."""
    run_config = config if config is not None else ForecastExportConfig()
    paths = build_project_paths()

    source_path = _resolve_output_path(
        default_path=paths.artifacts_dir / "forecasts_latest.csv",
        requested=run_config.input_path,
        project_paths=paths,
    )
    if not source_path.exists():
        raise FileNotFoundError(
            f"Forecast source file is missing: {source_path}. Run forecast-next first."
        )

    frame = pd.read_csv(source_path)
    required_columns = {DATE_COLUMN, STORE_COLUMN, PRODUCT_COLUMN, "forecast_units"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(
            "Forecast source file is missing required columns: " + ", ".join(missing)
        )

    output_path = _resolve_output_path(
        default_path=paths.artifacts_dir / "forecasts_latest.csv",
        requested=run_config.output_path,
        project_paths=paths,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    summary_payload: dict[str, object] = {
        "generated_at_utc": _utc_now_iso(),
        "source_path": str(source_path),
        "output_path": str(output_path),
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "columns": frame.columns.tolist(),
    }
    summary_path = write_json(summary_payload, paths.artifacts_dir / "forecast_export_summary.json")

    return {
        "forecasts_export_csv": output_path,
        "forecast_export_summary_json": summary_path,
    }


def _load_best_model_record(project_paths: ProjectPaths) -> dict[str, object]:
    registry_path = project_paths.artifacts_dir / "best_model_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Best model registry is missing: {registry_path}. Run evaluate-forecast-models first."
        )

    registry = pd.read_csv(registry_path)
    if registry.empty:
        raise ValueError("best_model_registry.csv is empty")

    overall = registry.loc[registry["scope"].astype("string") == "overall"].head(1)
    if overall.empty:
        raise ValueError("best_model_registry.csv does not contain an overall winner row")

    raw_record = overall.iloc[0].to_dict()
    return {str(key): value for key, value in raw_record.items()}


def _select_latest_split(bundle: ForecastingDataBundle) -> tuple[str, pd.DataFrame]:
    if not bundle.test_frame.empty:
        return "test", bundle.test_frame.copy(deep=True)
    if not bundle.validation_frame.empty:
        return "validation", bundle.validation_frame.copy(deep=True)
    if not bundle.train_frame.empty:
        return "train", bundle.train_frame.copy(deep=True)

    raise ValueError("No rows are available in any split for forecast generation")


def _resolve_output_path(
    default_path: Path,
    requested: Path | None,
    project_paths: ProjectPaths,
) -> Path:
    candidate = default_path if requested is None else Path(requested)
    if not candidate.is_absolute():
        candidate = project_paths.project_root / candidate
    return candidate


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()
