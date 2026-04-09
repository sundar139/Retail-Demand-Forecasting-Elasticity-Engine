"""Phase 6 Optuna tuning for forecast models with time-aware validation scoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from retail_forecasting.evaluation import compute_metric_value
from retail_forecasting.forecasting_models import (
    MODEL_ALL,
    MODEL_LIGHTGBM,
    MODEL_XGBOOST,
    DEFAULT_TARGET_COLUMN,
    build_forecast_model,
    load_forecasting_data_bundle,
    prepare_forecasting_matrices,
    resolve_model_list,
)
from retail_forecasting.paths import build_project_paths
from retail_forecasting.preprocessing import write_json

LOGGER = logging.getLogger(__name__)

optuna: Any
try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None


@dataclass(frozen=True, slots=True)
class ForecastTuningConfig:
    """Configuration for optional Optuna tuning workflows."""

    model: str = MODEL_ALL
    optimize_metric: str = "wmape"
    n_trials: int = 20
    random_state: int = 42
    target_column: str = DEFAULT_TARGET_COLUMN
    use_llm_features: bool = True
    include_price_features: bool = True
    input_path: Path | None = None
    llm_features_path: Path | None = None


def tune_forecast_models_pipeline(
    config: ForecastTuningConfig | None = None,
) -> dict[str, Path]:
    """Run bounded Optuna tuning for selected forecast model families."""
    if optuna is None:
        raise ImportError(
            "Optuna is not installed. Install project dependencies with "
            "`uv sync --all-groups --python 3.13`."
        )

    run_config = config if config is not None else ForecastTuningConfig()
    if run_config.n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if run_config.random_state < 0:
        raise ValueError("random_state must be non-negative")

    model_names = resolve_model_list(run_config.model)
    paths = build_project_paths()

    bundle = load_forecasting_data_bundle(
        target_column=run_config.target_column,
        use_llm_features=run_config.use_llm_features,
        input_path=run_config.input_path,
        llm_features_path=run_config.llm_features_path,
        project_paths=paths,
    )
    matrices = prepare_forecasting_matrices(
        bundle=bundle,
        include_price_features=run_config.include_price_features,
        include_identifier_encodings=True,
    )

    outputs: dict[str, Path] = {}
    warnings: list[str] = []
    summary_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []

    for model_name in model_names:
        try:
            sampler = optuna.samplers.TPESampler(seed=run_config.random_state)
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                study_name=f"phase6_{model_name}",
            )

            def objective(trial: Any) -> float:
                params = _suggest_params(model_name=model_name, trial=trial)
                wrapper = build_forecast_model(
                    model_name=model_name,
                    random_state=run_config.random_state,
                    params=params,
                )
                wrapper.fit(matrices.X_train, matrices.y_train)
                predictions = wrapper.predict(matrices.X_validation)
                value = compute_metric_value(
                    matrices.y_validation,
                    predictions,
                    metric_name=run_config.optimize_metric,
                )
                if not np.isfinite(value):
                    return 1e12
                return float(value)

            study.optimize(objective, n_trials=run_config.n_trials, show_progress_bar=False)

            best_payload = {
                "model_name": model_name,
                "optimize_metric": run_config.optimize_metric,
                "best_value": float(study.best_value),
                "best_params": dict(study.best_params),
                "n_trials": int(len(study.trials)),
                "completed_at_utc": _utc_now_iso(),
            }

            best_path = paths.artifacts_dir / f"optuna_{model_name}_best_params.json"
            write_json(best_payload, best_path)
            outputs[f"optuna_{model_name}_best_params_json"] = best_path

            summary_rows.append(
                {
                    "model_name": model_name,
                    "status": "success",
                    "optimize_metric": run_config.optimize_metric,
                    "best_value": float(study.best_value),
                    "n_trials": int(len(study.trials)),
                    "best_params_path": str(best_path),
                    "error": "",
                }
            )

            for trial in study.trials:
                trial_rows.append(
                    {
                        "model_name": model_name,
                        "trial_number": int(trial.number),
                        "state": str(trial.state),
                        "objective_value": (
                            float(trial.value)
                            if trial.value is not None and np.isfinite(float(trial.value))
                            else float("nan")
                        ),
                        "params_json": json.dumps(trial.params, sort_keys=True),
                    }
                )

        except Exception as exc:  # noqa: BLE001
            warning = f"Tuning failed for {model_name}: {exc}"
            warnings.append(warning)
            LOGGER.exception(warning)
            summary_rows.append(
                {
                    "model_name": model_name,
                    "status": "failed",
                    "optimize_metric": run_config.optimize_metric,
                    "best_value": float("nan"),
                    "n_trials": 0,
                    "best_params_path": "",
                    "error": str(exc),
                }
            )

    summary_path = paths.artifacts_dir / "optuna_study_summary.csv"
    pd.DataFrame.from_records(summary_rows).to_csv(summary_path, index=False)
    outputs["optuna_study_summary_csv"] = summary_path

    trials_path = paths.artifacts_dir / "optuna_trials_detailed.csv"
    pd.DataFrame.from_records(trial_rows).to_csv(trials_path, index=False)
    outputs["optuna_trials_detailed_csv"] = trials_path

    summary_json_payload: dict[str, object] = {
        "run_at_utc": _utc_now_iso(),
        "requested_models": model_names,
        "optimize_metric": run_config.optimize_metric,
        "n_trials_requested": int(run_config.n_trials),
        "warnings": warnings,
        "output_paths": {name: str(path) for name, path in outputs.items()},
    }
    summary_json_path = write_json(summary_json_payload, paths.artifacts_dir / "optuna_run_summary.json")
    outputs["optuna_run_summary_json"] = summary_json_path

    if not any(row.get("status") == "success" for row in summary_rows):
        raise RuntimeError("No model tuning study completed successfully")

    return outputs


def _suggest_params(model_name: str, trial: Any) -> dict[str, object]:
    normalized = model_name.strip().lower()

    if normalized == MODEL_LIGHTGBM:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 120, 420),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    if normalized == MODEL_XGBOOST:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 120, 420),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    raise ValueError(f"Unsupported model for tuning: {model_name}")


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()
