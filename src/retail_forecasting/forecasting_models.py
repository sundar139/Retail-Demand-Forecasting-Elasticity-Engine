"""Phase 6 forecasting data loading, model wrappers, and training pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import logging
from typing import Any, Protocol, cast

import joblib
import numpy as np
import pandas as pd

from retail_forecasting.features_common import ensure_group_date_sort_order
from retail_forecasting.llm_metadata import derive_llm_usage_facts, llm_usage_facts_to_dict
from retail_forecasting.paths import ProjectPaths, build_project_paths
from retail_forecasting.preprocessing import load_json, write_json
from retail_forecasting.schemas import DATE_COLUMN, PRICE_COLUMN, PRODUCT_COLUMN, STORE_COLUMN, UNITS_COLUMN

LOGGER = logging.getLogger(__name__)

_lightgbm_module: Any
try:
    import lightgbm as _lightgbm_module
except ImportError:  # pragma: no cover
    _lightgbm_module = None

_xgboost_module: Any
try:
    import xgboost as _xgboost_module
except ImportError:  # pragma: no cover
    _xgboost_module = None

MODEL_LIGHTGBM = "lightgbm"
MODEL_XGBOOST = "xgboost"
MODEL_ALL = "all"
ALLOWED_MODEL_SELECTION: frozenset[str] = frozenset({MODEL_LIGHTGBM, MODEL_XGBOOST, MODEL_ALL})

SEGMENT_MODE_GLOBAL = "global"
SEGMENT_MODE_PER_PRODUCT = "per-product"
ALLOWED_SEGMENT_MODES: frozenset[str] = frozenset({SEGMENT_MODE_GLOBAL, SEGMENT_MODE_PER_PRODUCT})

DEFAULT_TARGET_COLUMN = UNITS_COLUMN
DEFAULT_MERGE_KEYS: tuple[str, str, str] = (DATE_COLUMN, STORE_COLUMN, PRODUCT_COLUMN)

LEAKAGE_BLOCKLIST: frozenset[str] = frozenset(
    {
        DATE_COLUMN,
        STORE_COLUMN,
        PRODUCT_COLUMN,
        "y_log",
        "t_log",
        "forecast_error_hint",
        "forecast_units",
        "baseline_units",
        "planned_price",
        "reference_price",
        "applied_elasticity",
        "elasticity_estimate",
        "lower_ci",
        "upper_ci",
        "segment_key",
        "fit_status",
        "skip_reason",
        "actual",
        "prediction",
        "prediction_error",
        "prediction_abs_error",
    }
)

LEAKAGE_SUBSTRINGS: tuple[str, ...] = (
    "future",
    "target_",
    "leak",
    "_ahead",
)

OPTUNA_BEST_PARAM_FILES: dict[str, str] = {
    MODEL_LIGHTGBM: "optuna_lightgbm_best_params.json",
    MODEL_XGBOOST: "optuna_xgboost_best_params.json",
}


class SupportsPredict(Protocol):
    """Protocol for estimator objects exposing predict during inference."""

    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


class SupportsFitPredict(SupportsPredict, Protocol):
    """Protocol for estimator objects exposing fit and predict."""

    def fit(self, features: pd.DataFrame, target: pd.Series) -> object: ...


@dataclass(frozen=True, slots=True)
class ForecastingDataBundle:
    """Container for split-aware forecasting inputs and optional augmentation metadata."""

    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame
    target_column: str
    training_source: str
    source_data_path: Path
    llm_added_columns: list[str]
    notes: list[str]


@dataclass(frozen=True, slots=True)
class PreparedForecastMatrices:
    """Prepared matrices and metadata for fitting and scoring forecast models."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    feature_columns: list[str]
    feature_medians: dict[str, float]
    category_maps: dict[str, dict[str, int]]
    llm_feature_columns_in_matrix: list[str]
    llm_features_used: bool
    excluded_feature_reasons: dict[str, str]


@dataclass(frozen=True, slots=True)
class ForecastModelTrainingConfig:
    """Configuration for Phase 6 model training runs."""

    model: str = MODEL_ALL
    segment_mode: str = SEGMENT_MODE_GLOBAL
    target_column: str = DEFAULT_TARGET_COLUMN
    use_llm_features: bool = True
    include_price_features: bool = True
    random_state: int = 42
    use_tuned_params: bool = True
    input_path: Path | None = None
    llm_features_path: Path | None = None


@dataclass
class TrainedForecastModelArtifact:
    """Persisted model artifact with feature metadata for reliable inference."""

    model_name: str
    model_family: str
    segment_mode: str
    estimator: SupportsPredict | None
    estimators_by_product: dict[str, SupportsPredict] | None
    fallback_estimator: SupportsPredict | None
    feature_columns: list[str]
    feature_medians: dict[str, float]
    category_maps: dict[str, dict[str, int]]
    target_column: str
    training_source: str
    llm_features_used: bool
    llm_feature_columns_in_matrix: list[str]
    random_state: int

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Generate numeric predictions from a raw split dataframe."""
        matrix = _prepare_inference_matrix(
            frame=frame,
            feature_columns=self.feature_columns,
            feature_medians=self.feature_medians,
            category_maps=self.category_maps,
        )

        if self.segment_mode == SEGMENT_MODE_GLOBAL:
            if self.estimator is None:
                raise ValueError("Global model artifact is missing estimator")
            predictions = self.estimator.predict(matrix)
            return np.asarray(predictions, dtype="float64")

        if self.estimators_by_product is None:
            raise ValueError("Per-product model artifact is missing segment estimators")

        product_values = frame[PRODUCT_COLUMN].astype("string").fillna("missing")
        outputs = np.zeros(len(frame), dtype="float64")

        for product_id in product_values.unique().tolist():
            mask = product_values == product_id
            segment_index = mask.to_numpy(dtype=bool)
            segment_matrix = matrix.loc[segment_index]
            model = self.estimators_by_product.get(str(product_id), self.fallback_estimator)
            if model is None:
                raise ValueError(
                    "Per-product model artifact has no fallback estimator for unseen products"
                )
            outputs[segment_index] = np.asarray(model.predict(segment_matrix), dtype="float64")

        return outputs


class LightGBMForecaster:
    """Stable LightGBM regressor wrapper with unified fit/predict interface."""

    def __init__(
        self,
        random_state: int = 42,
        params: Mapping[str, object] | None = None,
    ) -> None:
        if _lightgbm_module is None:
            raise ImportError(
                "LightGBM is not installed. Install project dependencies with "
                "`uv sync --all-groups --python 3.13`."
            )

        default_params: dict[str, object] = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": random_state,
            "n_jobs": -1,
        }
        merged_params = {**default_params, **(dict(params) if params is not None else {})}
        lightgbm_module = cast(Any, _lightgbm_module)
        self._estimator = cast(SupportsFitPredict, lightgbm_module.LGBMRegressor(**merged_params))

    @property
    def estimator(self) -> SupportsPredict:
        return self._estimator

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        self._estimator.fit(features, target)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._estimator.predict(features), dtype="float64")


class XGBoostForecaster:
    """Stable XGBoost regressor wrapper with unified fit/predict interface."""

    def __init__(
        self,
        random_state: int = 42,
        params: Mapping[str, object] | None = None,
    ) -> None:
        if _xgboost_module is None:
            raise ImportError(
                "XGBoost is not installed. Install project dependencies with "
                "`uv sync --all-groups --python 3.13`."
            )

        default_params: dict[str, object] = {
            "n_estimators": 320,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,
        }
        merged_params = {**default_params, **(dict(params) if params is not None else {})}
        xgboost_module = cast(Any, _xgboost_module)
        self._estimator = cast(SupportsFitPredict, xgboost_module.XGBRegressor(**merged_params))

    @property
    def estimator(self) -> SupportsPredict:
        return self._estimator

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        self._estimator.fit(features, target)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._estimator.predict(features), dtype="float64")


def resolve_model_list(model: str) -> list[str]:
    """Resolve model selection flag into concrete model names."""
    normalized = model.strip().lower()
    if normalized not in ALLOWED_MODEL_SELECTION:
        allowed = ", ".join(sorted(ALLOWED_MODEL_SELECTION))
        raise ValueError(f"Unsupported model selection '{model}'. Allowed values: {allowed}")

    if normalized == MODEL_ALL:
        return [MODEL_LIGHTGBM, MODEL_XGBOOST]
    return [normalized]


def validate_segment_mode(segment_mode: str) -> str:
    """Validate segment mode option and return normalized value."""
    normalized = segment_mode.strip().lower()
    if normalized not in ALLOWED_SEGMENT_MODES:
        allowed = ", ".join(sorted(ALLOWED_SEGMENT_MODES))
        raise ValueError(f"Unsupported segment mode '{segment_mode}'. Allowed values: {allowed}")
    return normalized


def build_forecast_model(
    model_name: str,
    random_state: int = 42,
    params: Mapping[str, object] | None = None,
) -> LightGBMForecaster | XGBoostForecaster:
    """Build forecast model wrapper for requested model family."""
    normalized = model_name.strip().lower()
    if normalized == MODEL_LIGHTGBM:
        return LightGBMForecaster(random_state=random_state, params=params)
    if normalized == MODEL_XGBOOST:
        return XGBoostForecaster(random_state=random_state, params=params)

    allowed = ", ".join(sorted(ALLOWED_MODEL_SELECTION - {MODEL_ALL}))
    raise ValueError(f"Unsupported model '{model_name}'. Allowed values: {allowed}")


def load_forecasting_data_bundle(
    target_column: str = DEFAULT_TARGET_COLUMN,
    use_llm_features: bool = True,
    input_path: Path | None = None,
    llm_features_path: Path | None = None,
    project_paths: ProjectPaths | None = None,
) -> ForecastingDataBundle:
    """Load split-aware forecasting datasets with optional LLM feature augmentation."""
    paths = project_paths if project_paths is not None else build_project_paths()
    notes: list[str] = []

    source_data_path: Path
    split_source: str

    if input_path is None:
        split_artifacts = _try_load_split_feature_artifacts(paths, target_column=target_column)
    else:
        split_artifacts = None

    if split_artifacts is not None:
        train_frame, validation_frame, test_frame = split_artifacts
        source_data_path = paths.data_processed_dir
        split_source = "split_feature_artifacts"
    else:
        source_data_path, manual_frame = _load_manual_feature_source(
            input_path=input_path,
            project_paths=paths,
            target_column=target_column,
        )
        train_frame, validation_frame, test_frame, split_source = _split_manual_features(
            manual_frame=manual_frame,
            project_paths=paths,
        )
        notes.append(
            "Split-specific feature artifacts were unavailable; using manual features with split metadata fallback"
        )

    train_frame = ensure_group_date_sort_order(train_frame)
    validation_frame = ensure_group_date_sort_order(validation_frame)
    test_frame = ensure_group_date_sort_order(test_frame)

    _validate_required_columns(train_frame, target_column)
    _validate_required_columns(validation_frame, target_column)
    _validate_required_columns(test_frame, target_column)
    validate_chronological_split_integrity(train_frame, validation_frame, test_frame)

    llm_added_columns: list[str] = []
    if use_llm_features:
        train_frame, validation_frame, test_frame, llm_added_columns, llm_notes = _merge_optional_llm_features(
            train_frame=train_frame,
            validation_frame=validation_frame,
            test_frame=test_frame,
            project_paths=paths,
            llm_features_path=llm_features_path,
        )
        notes.extend(llm_notes)

    return ForecastingDataBundle(
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
        target_column=target_column,
        training_source=split_source,
        source_data_path=source_data_path,
        llm_added_columns=llm_added_columns,
        notes=notes,
    )


def validate_chronological_split_integrity(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> None:
    """Ensure chronological split ordering and non-overlap are preserved."""
    train_max = pd.Timestamp(train_frame[DATE_COLUMN].max())
    validation_min = pd.Timestamp(validation_frame[DATE_COLUMN].min())
    validation_max = pd.Timestamp(validation_frame[DATE_COLUMN].max())
    test_min = pd.Timestamp(test_frame[DATE_COLUMN].min())

    if train_max >= validation_min:
        raise ValueError("Chronology violation: train split overlaps validation period")
    if validation_max >= test_min:
        raise ValueError("Chronology violation: validation split overlaps test period")


def prepare_forecasting_matrices(
    bundle: ForecastingDataBundle,
    include_price_features: bool = True,
    include_identifier_encodings: bool = True,
) -> PreparedForecastMatrices:
    """Prepare consistent train/validation/test matrices with auditable feature selection."""
    train_frame = bundle.train_frame.copy(deep=True)
    validation_frame = bundle.validation_frame.copy(deep=True)
    test_frame = bundle.test_frame.copy(deep=True)

    category_maps: dict[str, dict[str, int]] = {}
    if include_identifier_encodings:
        train_frame, validation_frame, test_frame, category_maps = _apply_identifier_encodings(
            train_frame=train_frame,
            validation_frame=validation_frame,
            test_frame=test_frame,
        )

    feature_columns, excluded_reasons = select_model_feature_columns(
        train_frame=train_frame,
        target_column=bundle.target_column,
        include_price_features=include_price_features,
    )

    if not feature_columns:
        raise ValueError("No eligible model features remained after filtering rules")

    feature_medians = {
        column_name: float(
            pd.to_numeric(train_frame[column_name], errors="coerce").astype("float64").median(skipna=True)
        )
        for column_name in feature_columns
    }

    feature_medians = {
        column_name: (0.0 if np.isnan(median_value) else median_value)
        for column_name, median_value in feature_medians.items()
    }

    X_train = _build_feature_matrix(train_frame, feature_columns, feature_medians)
    X_validation = _build_feature_matrix(validation_frame, feature_columns, feature_medians)
    X_test = _build_feature_matrix(test_frame, feature_columns, feature_medians)

    y_train = pd.to_numeric(train_frame[bundle.target_column], errors="coerce").astype("float64")
    y_validation = pd.to_numeric(validation_frame[bundle.target_column], errors="coerce").astype("float64")
    y_test = pd.to_numeric(test_frame[bundle.target_column], errors="coerce").astype("float64")

    if y_train.isna().any() or y_validation.isna().any() or y_test.isna().any():
        raise ValueError("Target column contains non-numeric or missing values after coercion")

    llm_feature_columns_in_matrix = [
        column_name for column_name in bundle.llm_added_columns if column_name in feature_columns
    ]

    return PreparedForecastMatrices(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        feature_columns=feature_columns,
        feature_medians=feature_medians,
        category_maps=category_maps,
        llm_feature_columns_in_matrix=llm_feature_columns_in_matrix,
        llm_features_used=bool(llm_feature_columns_in_matrix),
        excluded_feature_reasons=excluded_reasons,
    )


def select_model_feature_columns(
    train_frame: pd.DataFrame,
    target_column: str,
    include_price_features: bool = True,
) -> tuple[list[str], dict[str, str]]:
    """Select numeric, leakage-safe model features from a training frame."""
    excluded_reasons: dict[str, str] = {}

    explicit_exclusions = set(LEAKAGE_BLOCKLIST)
    explicit_exclusions.add(target_column)

    if not include_price_features:
        explicit_exclusions.update({PRICE_COLUMN, "discount", "price_change_1d", "price_change_7d"})

    selected: list[str] = []

    for column_name in train_frame.columns:
        if column_name in explicit_exclusions:
            excluded_reasons[column_name] = "explicit_exclusion"
            continue

        lowered = column_name.lower()
        if any(token in lowered for token in LEAKAGE_SUBSTRINGS):
            excluded_reasons[column_name] = "leakage_pattern"
            continue

        if not pd.api.types.is_numeric_dtype(train_frame[column_name]):
            excluded_reasons[column_name] = "non_numeric"
            continue

        numeric = pd.to_numeric(train_frame[column_name], errors="coerce").astype("float64")
        if int(numeric.notna().sum()) < 2:
            excluded_reasons[column_name] = "insufficient_non_null"
            continue

        variance = float(numeric.var(ddof=0))
        if variance <= 0.0:
            excluded_reasons[column_name] = "zero_variance"
            continue

        selected.append(column_name)

    return selected, excluded_reasons


def load_optuna_best_params(
    model_name: str,
    project_paths: ProjectPaths | None = None,
) -> dict[str, object] | None:
    """Load best Optuna params when previous tuning artifacts are available."""
    paths = project_paths if project_paths is not None else build_project_paths()
    normalized = model_name.strip().lower()
    filename = OPTUNA_BEST_PARAM_FILES.get(normalized)
    if filename is None:
        return None

    target = paths.artifacts_dir / filename
    if not target.exists():
        return None

    payload = load_json(target)
    best_params = payload.get("best_params")
    if isinstance(best_params, dict):
        return dict(best_params)
    if isinstance(payload, dict):
        return dict(payload)
    return None


def train_forecast_models_pipeline(
    config: ForecastModelTrainingConfig | None = None,
) -> dict[str, Path]:
    """Train requested forecast models and persist reusable model artifacts."""
    run_config = config if config is not None else ForecastModelTrainingConfig()

    model_names = resolve_model_list(run_config.model)
    segment_mode = validate_segment_mode(run_config.segment_mode)
    if run_config.random_state < 0:
        raise ValueError("random_state must be non-negative")

    paths = build_project_paths()
    data_bundle = load_forecasting_data_bundle(
        target_column=run_config.target_column,
        use_llm_features=run_config.use_llm_features,
        input_path=run_config.input_path,
        llm_features_path=run_config.llm_features_path,
        project_paths=paths,
    )
    matrices = prepare_forecasting_matrices(
        data_bundle,
        include_price_features=run_config.include_price_features,
        include_identifier_encodings=True,
    )

    models_dir = paths.artifacts_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    warnings: list[str] = []
    records: list[dict[str, object]] = []

    run_llm_usage = derive_llm_usage_facts(
        llm_requested=run_config.use_llm_features,
        llm_feature_columns_used=matrices.llm_feature_columns_in_matrix,
        project_paths=paths,
    )

    for model_name in model_names:
        tuned_params = (
            load_optuna_best_params(model_name=model_name, project_paths=paths)
            if run_config.use_tuned_params
            else None
        )

        try:
            artifact = _fit_model_artifact(
                model_name=model_name,
                segment_mode=segment_mode,
                matrices=matrices,
                train_frame=data_bundle.train_frame,
                target_column=run_config.target_column,
                training_source=data_bundle.training_source,
                random_state=run_config.random_state,
                tuned_params=tuned_params,
            )
            llm_columns_used = [
                column_name
                for column_name in artifact.llm_feature_columns_in_matrix
                if column_name in artifact.feature_columns
            ]
            model_llm_usage = derive_llm_usage_facts(
                llm_requested=run_config.use_llm_features,
                llm_feature_columns_used=llm_columns_used,
                project_paths=paths,
            )

            if model_llm_usage.llm_features_actually_used:
                artifact.llm_feature_columns_in_matrix = model_llm_usage.llm_feature_columns_used
                artifact.llm_features_used = True
            else:
                artifact.llm_feature_columns_in_matrix = []
                artifact.llm_features_used = False

            model_path = models_dir / f"{model_name}_{segment_mode}.joblib"
            joblib.dump(artifact, model_path)
            outputs[f"{model_name}_model_joblib"] = model_path

            records.append(
                {
                    "trained_at_utc": _utc_now_iso(),
                    "model_name": model_name,
                    "model_family": model_name,
                    "segment_mode": segment_mode,
                    "status": "success",
                    "error": "",
                    "model_path": str(model_path),
                    "training_source": data_bundle.training_source,
                    "feature_count": int(len(artifact.feature_columns)),
                    "training_rows": int(len(data_bundle.train_frame)),
                    "llm_features_used": bool(model_llm_usage.llm_features_actually_used),
                    "llm_feature_count": int(len(artifact.llm_feature_columns_in_matrix)),
                    "llm_feature_columns": "|".join(artifact.llm_feature_columns_in_matrix),
                    "used_tuned_params": bool(tuned_params),
                }
            )
        except Exception as exc:  # noqa: BLE001
            warning = f"Model training failed for {model_name}: {exc}"
            warnings.append(warning)
            LOGGER.exception(warning)
            records.append(
                {
                    "trained_at_utc": _utc_now_iso(),
                    "model_name": model_name,
                    "model_family": model_name,
                    "segment_mode": segment_mode,
                    "status": "failed",
                    "error": str(exc),
                    "model_path": "",
                    "training_source": data_bundle.training_source,
                    "feature_count": 0,
                    "training_rows": int(len(data_bundle.train_frame)),
                    "llm_features_used": False,
                    "llm_feature_count": 0,
                    "llm_feature_columns": "",
                    "used_tuned_params": bool(tuned_params),
                }
            )

    registry_frame = pd.DataFrame.from_records(records)
    registry_path = paths.artifacts_dir / "model_training_registry.csv"
    registry_frame.to_csv(registry_path, index=False)
    outputs["model_training_registry_csv"] = registry_path

    summary_payload: dict[str, object] = {
        "trained_at_utc": _utc_now_iso(),
        "requested_models": model_names,
        "segment_mode": segment_mode,
        "target_column": run_config.target_column,
        "training_source": data_bundle.training_source,
        "feature_count": int(len(matrices.feature_columns)),
        "llm_requested": bool(run_config.use_llm_features),
        "llm_flag_requested": bool(run_config.use_llm_features),
        "ollama_reachable": run_llm_usage.ollama_reachable,
        "planner_model_available": run_llm_usage.planner_model_available,
        "llm_feature_file_exists": run_llm_usage.llm_feature_file_exists,
        "llm_feature_file_path": run_llm_usage.llm_feature_file_path,
        "llm_output_feature_count": int(run_llm_usage.llm_output_feature_count),
        "llm_features_used": bool(run_llm_usage.llm_features_actually_used),
        "llm_features_actually_used": bool(run_llm_usage.llm_features_actually_used),
        "llm_feature_columns_used": (
            run_llm_usage.llm_feature_columns_used if run_llm_usage.llm_features_actually_used else []
        ),
        "llm_usage_facts": llm_usage_facts_to_dict(run_llm_usage),
        "warnings": warnings,
        "output_paths": {name: str(path) for name, path in outputs.items()},
    }

    summary_path = write_json(summary_payload, paths.artifacts_dir / "model_training_summary.json")
    outputs["model_training_summary_json"] = summary_path

    successful_models = [record for record in records if record.get("status") == "success"]
    if not successful_models:
        raise RuntimeError(
            "No models were successfully trained. Check model_training_registry.csv for errors."
        )

    return outputs


def load_model_training_registry(
    registry_path: Path | None = None,
    project_paths: ProjectPaths | None = None,
) -> pd.DataFrame:
    """Load model training registry generated by train-forecast-models."""
    paths = project_paths if project_paths is not None else build_project_paths()
    target = paths.artifacts_dir / "model_training_registry.csv" if registry_path is None else Path(registry_path)
    if not target.is_absolute():
        target = paths.project_root / target

    if not target.exists():
        raise FileNotFoundError(
            f"Missing model training registry: {target}. Run 'train-forecast-models' first."
        )

    return pd.read_csv(target)


def load_trained_model_artifacts(
    model: str = MODEL_ALL,
    segment_mode: str | None = None,
    registry_path: Path | None = None,
    project_paths: ProjectPaths | None = None,
) -> dict[str, TrainedForecastModelArtifact]:
    """Load trained model artifacts from latest training registry entries."""
    registry = load_model_training_registry(registry_path=registry_path, project_paths=project_paths)
    requested_models = resolve_model_list(model)

    successful = registry.loc[registry["status"].astype("string") == "success"].copy()
    if successful.empty:
        return {}

    successful = successful.loc[
        successful["model_name"].astype("string").isin(requested_models)
    ].copy()

    if segment_mode is not None:
        normalized_mode = validate_segment_mode(segment_mode)
        successful = successful.loc[
            successful["segment_mode"].astype("string") == normalized_mode
        ].copy()

    if successful.empty:
        return {}

    successful = successful.sort_values(by=["trained_at_utc"], kind="mergesort")
    successful = successful.drop_duplicates(subset=["model_name"], keep="last")

    artifacts: dict[str, TrainedForecastModelArtifact] = {}
    for row in successful.itertuples(index=False):
        model_name = str(getattr(row, "model_name"))
        model_path = Path(str(getattr(row, "model_path")))
        if not model_path.exists():
            LOGGER.warning("Skipping missing model artifact for %s at %s", model_name, model_path)
            continue

        loaded = joblib.load(model_path)
        if isinstance(loaded, TrainedForecastModelArtifact):
            loaded.llm_feature_columns_in_matrix = [
                column_name
                for column_name in loaded.llm_feature_columns_in_matrix
                if column_name in loaded.feature_columns
            ]
            loaded.llm_features_used = bool(loaded.llm_feature_columns_in_matrix)
            artifacts[model_name] = loaded
            continue

        raise ValueError(
            f"Unexpected model artifact type for {model_name} at {model_path}: {type(loaded)}"
        )

    return artifacts


def _fit_model_artifact(
    model_name: str,
    segment_mode: str,
    matrices: PreparedForecastMatrices,
    train_frame: pd.DataFrame,
    target_column: str,
    training_source: str,
    random_state: int,
    tuned_params: Mapping[str, object] | None,
) -> TrainedForecastModelArtifact:
    if segment_mode == SEGMENT_MODE_GLOBAL:
        wrapper = build_forecast_model(model_name, random_state=random_state, params=tuned_params)
        wrapper.fit(matrices.X_train, matrices.y_train)
        llm_columns_used = [
            column_name
            for column_name in matrices.llm_feature_columns_in_matrix
            if column_name in matrices.feature_columns
        ]
        return TrainedForecastModelArtifact(
            model_name=model_name,
            model_family=model_name,
            segment_mode=segment_mode,
            estimator=wrapper.estimator,
            estimators_by_product=None,
            fallback_estimator=None,
            feature_columns=matrices.feature_columns,
            feature_medians=matrices.feature_medians,
            category_maps=matrices.category_maps,
            target_column=target_column,
            training_source=training_source,
            llm_features_used=bool(llm_columns_used),
            llm_feature_columns_in_matrix=llm_columns_used,
            random_state=random_state,
        )

    estimators_by_product: dict[str, SupportsPredict] = {}
    grouped = train_frame.groupby(PRODUCT_COLUMN, sort=True, dropna=False)

    for raw_product_id, product_frame in grouped:
        product_id = str(raw_product_id)
        product_index = product_frame.index
        if len(product_index) < 5:
            continue

        wrapper = build_forecast_model(model_name, random_state=random_state, params=tuned_params)
        wrapper.fit(matrices.X_train.loc[product_index], matrices.y_train.loc[product_index])
        estimators_by_product[product_id] = wrapper.estimator

    if not estimators_by_product:
        raise ValueError(
            "Per-product mode did not produce any eligible segment models (insufficient rows per product)"
        )

    fallback_wrapper = build_forecast_model(model_name, random_state=random_state, params=tuned_params)
    fallback_wrapper.fit(matrices.X_train, matrices.y_train)

    llm_columns_used = [
        column_name
        for column_name in matrices.llm_feature_columns_in_matrix
        if column_name in matrices.feature_columns
    ]

    return TrainedForecastModelArtifact(
        model_name=model_name,
        model_family=model_name,
        segment_mode=segment_mode,
        estimator=None,
        estimators_by_product=estimators_by_product,
        fallback_estimator=fallback_wrapper.estimator,
        feature_columns=matrices.feature_columns,
        feature_medians=matrices.feature_medians,
        category_maps=matrices.category_maps,
        target_column=target_column,
        training_source=training_source,
        llm_features_used=bool(llm_columns_used),
        llm_feature_columns_in_matrix=llm_columns_used,
        random_state=random_state,
    )


def _apply_identifier_encodings(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
    store_values = sorted(train_frame[STORE_COLUMN].astype("string").dropna().unique().tolist())
    product_values = sorted(train_frame[PRODUCT_COLUMN].astype("string").dropna().unique().tolist())

    store_map = {str(value): index for index, value in enumerate(store_values)}
    product_map = {str(value): index for index, value in enumerate(product_values)}

    category_maps = {
        STORE_COLUMN: store_map,
        PRODUCT_COLUMN: product_map,
    }

    def encode(frame: pd.DataFrame) -> pd.DataFrame:
        encoded = frame.copy(deep=True)
        encoded["store_id_encoded"] = (
            encoded[STORE_COLUMN].astype("string").map(store_map).fillna(-1).astype("float64")
        )
        encoded["product_id_encoded"] = (
            encoded[PRODUCT_COLUMN].astype("string").map(product_map).fillna(-1).astype("float64")
        )
        return encoded

    return encode(train_frame), encode(validation_frame), encode(test_frame), category_maps


def _build_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    feature_medians: Mapping[str, float],
) -> pd.DataFrame:
    matrix = pd.DataFrame(index=frame.index)
    for column_name in feature_columns:
        values = pd.to_numeric(frame[column_name], errors="coerce").astype("float64")
        fill_value = float(feature_medians.get(column_name, 0.0))
        matrix[column_name] = values.fillna(fill_value)
    return matrix


def _prepare_inference_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    feature_medians: Mapping[str, float],
    category_maps: Mapping[str, Mapping[str, int]],
) -> pd.DataFrame:
    working = frame.copy(deep=True)

    if STORE_COLUMN in category_maps:
        working["store_id_encoded"] = (
            working[STORE_COLUMN]
            .astype("string")
            .map(dict(category_maps[STORE_COLUMN]))
            .fillna(-1)
            .astype("float64")
        )

    if PRODUCT_COLUMN in category_maps:
        working["product_id_encoded"] = (
            working[PRODUCT_COLUMN]
            .astype("string")
            .map(dict(category_maps[PRODUCT_COLUMN]))
            .fillna(-1)
            .astype("float64")
        )

    for column_name in feature_columns:
        if column_name not in working.columns:
            working[column_name] = np.nan

    return _build_feature_matrix(working, feature_columns=feature_columns, feature_medians=feature_medians)


def _validate_required_columns(frame: pd.DataFrame, target_column: str) -> None:
    required = {DATE_COLUMN, STORE_COLUMN, PRODUCT_COLUMN, target_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Forecasting dataframe is missing required columns: {', '.join(missing)}")


def _try_load_split_feature_artifacts(
    project_paths: ProjectPaths,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    train_path = project_paths.data_processed_dir / "features_train.parquet"
    validation_path = project_paths.data_processed_dir / "features_validation.parquet"
    test_path = project_paths.data_processed_dir / "features_test.parquet"

    if not (train_path.exists() and validation_path.exists() and test_path.exists()):
        return None

    train_frame = pd.read_parquet(train_path)
    validation_frame = pd.read_parquet(validation_path)
    test_frame = pd.read_parquet(test_path)

    if train_frame.empty or validation_frame.empty or test_frame.empty:
        return None

    _validate_required_columns(train_frame, target_column)
    _validate_required_columns(validation_frame, target_column)
    _validate_required_columns(test_frame, target_column)
    return train_frame, validation_frame, test_frame


def _load_manual_feature_source(
    input_path: Path | None,
    project_paths: ProjectPaths,
    target_column: str,
) -> tuple[Path, pd.DataFrame]:
    source_path = (
        project_paths.data_processed_dir / "features_manual.parquet"
        if input_path is None
        else Path(input_path).expanduser()
    )
    if not source_path.is_absolute():
        source_path = project_paths.project_root / source_path

    if not source_path.exists():
        raise FileNotFoundError(
            f"Manual feature source not found at {source_path}. "
            "Run 'build-manual-features' first."
        )

    frame = pd.read_parquet(source_path)
    _validate_required_columns(frame, target_column)
    return source_path, frame


def _split_manual_features(
    manual_frame: pd.DataFrame,
    project_paths: ProjectPaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    working = ensure_group_date_sort_order(manual_frame)

    split_summary_path = project_paths.artifacts_dir / "split_summary.json"
    if split_summary_path.exists():
        payload = load_json(split_summary_path)
        cutoffs = payload.get("cutoffs")
        if not isinstance(cutoffs, dict):
            raise ValueError("split_summary.json does not contain a valid 'cutoffs' object")

        validation_start = cutoffs.get("validation_start")
        test_start = cutoffs.get("test_start")
        if not isinstance(validation_start, str) or not isinstance(test_start, str):
            raise ValueError("split_summary.json is missing validation_start or test_start cutoffs")

        validation_start_ts = pd.to_datetime(validation_start, errors="raise").floor("D")
        test_start_ts = pd.to_datetime(test_start, errors="raise").floor("D")

        train_frame = working.loc[working[DATE_COLUMN] < validation_start_ts].reset_index(drop=True)
        validation_frame = working.loc[
            (working[DATE_COLUMN] >= validation_start_ts) & (working[DATE_COLUMN] < test_start_ts)
        ].reset_index(drop=True)
        test_frame = working.loc[working[DATE_COLUMN] >= test_start_ts].reset_index(drop=True)

        if train_frame.empty or validation_frame.empty or test_frame.empty:
            raise ValueError(
                "Manual feature fallback produced an empty split with split_summary cutoffs"
            )

        return train_frame, validation_frame, test_frame, "manual_features_plus_split_summary"

    train_raw = project_paths.data_processed_dir / "train.parquet"
    validation_raw = project_paths.data_processed_dir / "validation.parquet"
    test_raw = project_paths.data_processed_dir / "test.parquet"

    if not (train_raw.exists() and validation_raw.exists() and test_raw.exists()):
        raise FileNotFoundError(
            "Split-aware feature artifacts are missing and fallback metadata is unavailable. "
            "Expected either features_train/features_validation/features_test or split_summary.json."
        )

    validation_dates = pd.read_parquet(validation_raw)[DATE_COLUMN]
    test_dates = pd.read_parquet(test_raw)[DATE_COLUMN]

    validation_start_ts = pd.to_datetime(validation_dates, errors="coerce").min()
    test_start_ts = pd.to_datetime(test_dates, errors="coerce").min()

    if pd.isna(validation_start_ts) or pd.isna(test_start_ts):
        raise ValueError("Unable to infer split cutoffs from processed split parquet files")

    validation_start_ts = pd.Timestamp(validation_start_ts).floor("D")
    test_start_ts = pd.Timestamp(test_start_ts).floor("D")

    train_frame = working.loc[working[DATE_COLUMN] < validation_start_ts].reset_index(drop=True)
    validation_frame = working.loc[
        (working[DATE_COLUMN] >= validation_start_ts) & (working[DATE_COLUMN] < test_start_ts)
    ].reset_index(drop=True)
    test_frame = working.loc[working[DATE_COLUMN] >= test_start_ts].reset_index(drop=True)

    if train_frame.empty or validation_frame.empty or test_frame.empty:
        raise ValueError("Manual feature fallback produced an empty split from processed split metadata")

    return train_frame, validation_frame, test_frame, "manual_features_plus_processed_split_dates"


def _merge_optional_llm_features(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    project_paths: ProjectPaths,
    llm_features_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    notes: list[str] = []

    llm_path = (
        project_paths.data_processed_dir / "features_llm.parquet"
        if llm_features_path is None
        else Path(llm_features_path).expanduser()
    )
    if not llm_path.is_absolute():
        llm_path = project_paths.project_root / llm_path

    if not llm_path.exists():
        notes.append(f"LLM feature augmentation skipped: file not found at {llm_path}")
        return train_frame, validation_frame, test_frame, [], notes

    llm_frame = pd.read_parquet(llm_path)
    if llm_frame.empty:
        notes.append("LLM feature augmentation skipped: feature file is empty")
        return train_frame, validation_frame, test_frame, [], notes

    if not set(DEFAULT_MERGE_KEYS).issubset(llm_frame.columns):
        notes.append("LLM feature augmentation skipped: merge keys are missing in LLM feature file")
        return train_frame, validation_frame, test_frame, [], notes

    validated_feature_names, validation_notes = _load_validated_llm_feature_names(project_paths)
    notes.extend(validation_notes)
    if validated_feature_names is not None and not validated_feature_names:
        return train_frame, validation_frame, test_frame, [], notes

    llm_frame = ensure_group_date_sort_order(llm_frame)
    candidate_columns = [
        column_name
        for column_name in llm_frame.columns
        if column_name not in DEFAULT_MERGE_KEYS and column_name not in train_frame.columns
    ]

    if validated_feature_names is not None:
        allowed = set(validated_feature_names)
        candidate_columns = [
            column_name for column_name in candidate_columns if column_name in allowed
        ]
        if not candidate_columns:
            notes.append(
                "LLM feature augmentation skipped: no validated materialized LLM columns matched merge candidates"
            )
            return train_frame, validation_frame, test_frame, [], notes

    usable_columns = [
        column_name
        for column_name in candidate_columns
        if int(llm_frame[column_name].notna().sum()) > 0
    ]

    if not usable_columns:
        notes.append(
            "LLM feature augmentation skipped: no usable non-overlapping LLM feature columns found"
        )
        return train_frame, validation_frame, test_frame, [], notes

    llm_subset = llm_frame[list(DEFAULT_MERGE_KEYS) + usable_columns].drop_duplicates(
        subset=list(DEFAULT_MERGE_KEYS),
        keep="last",
    )

    merged_train = _merge_preserving_row_order(train_frame, llm_subset)
    merged_validation = _merge_preserving_row_order(validation_frame, llm_subset)
    merged_test = _merge_preserving_row_order(test_frame, llm_subset)

    notes.append(f"LLM feature augmentation merged {len(usable_columns)} columns")
    return merged_train, merged_validation, merged_test, usable_columns, notes


def _merge_preserving_row_order(base_frame: pd.DataFrame, llm_subset: pd.DataFrame) -> pd.DataFrame:
    working = base_frame.reset_index(drop=False).rename(columns={"index": "_row_order"})
    merged = working.merge(
        llm_subset,
        on=list(DEFAULT_MERGE_KEYS),
        how="left",
        sort=False,
        validate="m:1",
    )
    return (
        merged.sort_values(by=["_row_order"], kind="mergesort")
        .drop(columns=["_row_order"])
        .reset_index(drop=True)
    )


def _load_validated_llm_feature_names(
    project_paths: ProjectPaths,
) -> tuple[list[str] | None, list[str]]:
    summary_path = project_paths.artifacts_dir / "llm_features_summary.json"
    if not summary_path.exists():
        return None, []

    notes: list[str] = []
    payload = load_json(summary_path)
    output_names = payload.get("output_feature_names")
    output_count = payload.get("output_feature_count")

    if not isinstance(output_names, list):
        notes.append(
            "LLM feature summary is present but output_feature_names is invalid; skipping LLM augmentation"
        )
        return [], notes

    normalized_names = [str(item) for item in output_names if isinstance(item, str)]

    count_value = int(output_count) if isinstance(output_count, int) else len(normalized_names)
    if count_value <= 0 or not normalized_names:
        notes.append(
            "LLM feature augmentation skipped: summary reports zero validated materialized LLM features"
        )
        return [], notes

    return normalized_names, notes


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()
