"""Segmented own-price elasticity estimation with EconML LinearDML."""

from dataclasses import dataclass
from pathlib import Path
import logging
import warnings as py_warnings
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from retail_forecasting.causal_utils import (
    CAUSAL_FEATURE_PROFILE_LEAN,
    SEGMENT_LEVEL_PRODUCT,
    add_log_outcome_treatment,
    evaluate_segment_guardrails,
    format_segment_key,
    generate_elasticity_report_artifacts,
    load_causal_feature_frame,
    make_nuisance_models,
    prepare_segment_controls,
    resolve_segment_columns,
    select_causal_control_features,
    validate_causal_feature_profile,
)
from retail_forecasting.paths import build_project_paths
from retail_forecasting.preprocessing import load_json, write_json
from retail_forecasting.schemas import DATE_COLUMN, PRICE_COLUMN

LOGGER = logging.getLogger(__name__)

_INFERENCE_WARNING_TOKENS: tuple[str, ...] = (
    "co-variance matrix is underdetermined",
    "covariance matrix is underdetermined",
    "inference will be invalid",
    "inference",
    "singular",
)

try:
    from econml.dml import LinearDML
except ImportError:  # pragma: no cover - exercised only when optional dependency missing.
    LinearDML = None


@dataclass(frozen=True, slots=True)
class ElasticityRunConfig:
    """Configuration for segmented causal elasticity estimation."""

    segment_level: str = SEGMENT_LEVEL_PRODUCT
    feature_profile: str = CAUSAL_FEATURE_PROFILE_LEAN
    min_samples: int = 250
    min_non_null_pairs: int | None = None
    min_unique_price_values: int = 8
    min_log_price_std: float = 0.01
    min_log_units_std: float = 0.01
    epsilon: float = 1e-3
    use_llm_features: bool = True
    include_llm_controls: bool = False
    allow_demand_forecast_controls: bool = False
    include_store_indicators: bool = True
    nuisance_model: str = "random-forest"
    input_path: Path | None = None
    llm_features_path: Path | None = None
    alpha: float = 0.05
    min_variance: float = 1e-8
    random_state: int = 42
    top_n_segments: int = 10
    max_ci_plot_segments: int = 20


def fit_elasticity_pipeline(config: ElasticityRunConfig | None = None) -> dict[str, Path]:
    """Fit segmented own-price elasticity models and persist artifacts."""
    if LinearDML is None:
        raise ImportError(
            "EconML is not installed. Install project dependencies with `uv sync --all-groups --python 3.13`."
        )

    run_config = config if config is not None else ElasticityRunConfig()
    _validate_run_config(run_config)

    paths = build_project_paths()
    loaded = load_causal_feature_frame(
        input_path=run_config.input_path,
        use_llm_features=run_config.use_llm_features,
        llm_features_path=run_config.llm_features_path,
        project_paths=paths,
    )

    frame = add_log_outcome_treatment(loaded.frame, epsilon=run_config.epsilon)
    frame = frame.sort_values(by=["store_id", "product_id", DATE_COLUMN], kind="mergesort").reset_index(drop=True)

    segment_columns = resolve_segment_columns(run_config.segment_level)
    control_columns = select_causal_control_features(
        frame,
        feature_profile=run_config.feature_profile,
        treatment_column=PRICE_COLUMN,
        include_store_indicators=run_config.include_store_indicators,
        include_llm_derived=run_config.include_llm_controls,
        allow_demand_forecast=run_config.allow_demand_forecast_controls,
        extra_excluded_columns=["y_log", "t_log"],
    )

    records: list[dict[str, object]] = []
    warnings: list[str] = list(loaded.notes)
    nuisance_model_label = run_config.nuisance_model

    grouped = frame.groupby(list(segment_columns), sort=True, dropna=False)
    for raw_segment_key, segment_frame in grouped:
        ordered_segment = segment_frame.sort_values(DATE_COLUMN, kind="mergesort").reset_index(drop=True)
        segment_key = format_segment_key(segment_columns, raw_segment_key)

        guardrail = evaluate_segment_guardrails(
            ordered_segment,
            outcome_column="y_log",
            treatment_column="t_log",
            min_samples=run_config.min_samples,
            min_non_null_pairs=run_config.min_non_null_pairs,
            min_unique_treatment_values=run_config.min_unique_price_values,
            min_treatment_std=run_config.min_log_price_std,
            min_outcome_std=run_config.min_log_units_std,
            min_variance=run_config.min_variance,
        )

        base_record = {
            "segmentation_level": run_config.segment_level,
            "feature_profile_used": run_config.feature_profile,
            "segment_key": segment_key,
            "sample_size": int(guardrail.sample_size),
            "non_null_pair_count": int(guardrail.non_null_pair_count),
            "unique_price_values": int(guardrail.unique_treatment_values),
            "log_price_std": float(guardrail.treatment_std),
            "log_units_std": float(guardrail.outcome_std),
            "treatment_variance": float(guardrail.treatment_variance),
            "outcome_variance": float(guardrail.outcome_variance),
            "elasticity_estimate": np.nan,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "fit_status": "skipped",
            "quality_status": "skipped",
            "ci_caution_flag": False,
            "skip_reason": "",
            "feature_count_used": 0,
            "warning_count": 0,
            "inference_warning_count": 0,
            "warnings_text": "",
        }

        if not guardrail.is_valid:
            base_record["skip_reason"] = str(guardrail.reason)
            records.append(base_record)
            continue

        controls_matrix, controls_used = prepare_segment_controls(ordered_segment, control_columns)
        if controls_matrix.empty:
            base_record["skip_reason"] = "no_usable_controls_after_segment_filtering"
            records.append(base_record)
            continue

        y_values = ordered_segment["y_log"].to_numpy(dtype=float)
        t_values = ordered_segment["t_log"].to_numpy(dtype=float)

        try:
            model_y, model_t, nuisance_model_label = make_nuisance_models(
                run_config.nuisance_model,
                random_state=run_config.random_state,
            )

            linear_dml_cls = cast(Any, LinearDML)
            dml = linear_dml_cls(
                model_y=model_y,
                model_t=model_t,
                cv=KFold(n_splits=2, shuffle=False),
                random_state=run_config.random_state,
            )
            x_values = controls_matrix.to_numpy(dtype=float)
            with py_warnings.catch_warnings(record=True) as captured_warnings:
                py_warnings.simplefilter("always")
                dml.fit(Y=y_values, T=t_values, X=x_values)

                point_estimate = _estimate_segment_elasticity(dml, x_values)
                lower_ci, upper_ci = _estimate_segment_confidence_interval(
                    dml,
                    x_values,
                    alpha=run_config.alpha,
                )

            warning_messages = _normalize_warning_messages(captured_warnings)
            inference_warning_messages = [
                message for message in warning_messages if _is_inference_warning(message)
            ]

            quality_status = "ok"
            if inference_warning_messages or np.isnan(lower_ci) or np.isnan(upper_ci):
                quality_status = "warning_inference_unstable"

            records.append(
                {
                    **base_record,
                    "elasticity_estimate": point_estimate,
                    "lower_ci": lower_ci,
                    "upper_ci": upper_ci,
                    "fit_status": "success",
                    "quality_status": quality_status,
                    "ci_caution_flag": bool(quality_status == "warning_inference_unstable"),
                    "skip_reason": "",
                    "feature_count_used": int(len(controls_used)),
                    "warning_count": int(len(warning_messages)),
                    "inference_warning_count": int(len(inference_warning_messages)),
                    "warnings_text": " | ".join(warning_messages),
                }
            )
            for warning_text in inference_warning_messages:
                warnings.append(
                    f"Segment {segment_key} inference warning observed: {warning_text}"
                )
        except Exception as exc:  # noqa: BLE001
            base_record["fit_status"] = "failed"
            base_record["quality_status"] = "failed"
            base_record["skip_reason"] = f"fit_failed: {exc}"
            records.append(base_record)
            warnings.append(f"Segment {segment_key} failed during model fit: {exc}")

    estimates = pd.DataFrame.from_records(records)
    estimates = estimates.sort_values(by=["fit_status", "segment_key"], kind="mergesort").reset_index(drop=True)

    attempts = int(len(estimates))
    successful = int((estimates["fit_status"] == "success").sum())
    skipped = int((estimates["fit_status"] == "skipped").sum())
    failed = int((estimates["fit_status"] == "failed").sum())
    quality_counts = estimates["quality_status"].astype("string").value_counts(dropna=False)
    inference_warning_count = int(
        pd.to_numeric(estimates["inference_warning_count"], errors="coerce")
        .fillna(0)
        .sum()
    )
    ci_caution_present = bool(
        pd.Series(estimates["ci_caution_flag"], dtype="boolean").fillna(False).any()
    )

    estimates_path = paths.artifacts_dir / "elasticity_estimates.csv"
    estimates.to_csv(estimates_path, index=False)

    report_paths = generate_elasticity_report_artifacts(
        estimates,
        project_paths=paths,
        top_n=run_config.top_n_segments,
        max_ci_segments=run_config.max_ci_plot_segments,
    )

    summary_payload: dict[str, object] = {
        "source_data_path": str(loaded.source_data_path),
        "llm_feature_augmentation_used": bool(loaded.llm_augmentation_used),
        "llm_added_column_count": int(len(loaded.llm_added_columns)),
        "total_segments_attempted": attempts,
        "successful_fits": successful,
        "skipped_fits": skipped,
        "failed_fits": failed,
        "quality_status_counts": {str(key): int(value) for key, value in quality_counts.items()},
        "inference_warning_count": inference_warning_count,
        "inference_warnings_present": bool(inference_warning_count > 0),
        "ci_caution_present": ci_caution_present,
        "segmentation_level": run_config.segment_level,
        "feature_profile_used": run_config.feature_profile,
        "min_sample_threshold": int(run_config.min_samples),
        "min_non_null_pair_threshold": int(
            run_config.min_samples if run_config.min_non_null_pairs is None else run_config.min_non_null_pairs
        ),
        "min_unique_price_values_threshold": int(run_config.min_unique_price_values),
        "min_log_price_std_threshold": float(run_config.min_log_price_std),
        "min_log_units_std_threshold": float(run_config.min_log_units_std),
        "epsilon_used": float(run_config.epsilon),
        "nuisance_model_choice": nuisance_model_label,
        "notes_warnings": warnings,
        "control_feature_count": int(len(control_columns)),
        "control_features_used": control_columns,
        "leakage_safety_statement": (
            "Segments are sorted chronologically; log transforms are safe with positive epsilon; "
            "treatment and outcome are current-row logs while controls use a dedicated causal feature profile "
            "that excludes identifiers, own-price lag/rolling proxies, and known leakage-prone columns."
        ),
        "output_paths": {
            "elasticity_estimates_csv": str(estimates_path),
            "elasticity_run_summary_json": str(paths.artifacts_dir / "elasticity_run_summary.json"),
            **{key: str(value) for key, value in report_paths.items()},
        },
    }

    summary_path = write_json(summary_payload, paths.artifacts_dir / "elasticity_run_summary.json")

    outputs: dict[str, Path] = {
        "elasticity_estimates_csv": estimates_path,
        "elasticity_run_summary_json": summary_path,
        **report_paths,
    }

    LOGGER.info(
        "Elasticity fitting completed: attempted=%d successful=%d skipped=%d",
        attempts,
        successful,
        skipped,
    )
    return outputs


def load_elasticity_run_summary(summary_path: Path | None = None) -> dict[str, object]:
    """Load elasticity run summary JSON."""
    paths = build_project_paths()
    target = paths.artifacts_dir / "elasticity_run_summary.json" if summary_path is None else Path(summary_path)
    if not target.is_absolute():
        target = paths.project_root / target

    if not target.exists():
        raise FileNotFoundError(
            f"Missing elasticity summary artifact: {target}. Run 'fit-elasticity' first."
        )

    return load_json(target)


def load_elasticity_estimates(estimates_path: Path | None = None) -> pd.DataFrame:
    """Load elasticity estimates CSV artifact."""
    paths = build_project_paths()
    target = paths.artifacts_dir / "elasticity_estimates.csv" if estimates_path is None else Path(estimates_path)
    if not target.is_absolute():
        target = paths.project_root / target

    if not target.exists():
        raise FileNotFoundError(
            f"Missing elasticity estimates artifact: {target}. Run 'fit-elasticity' first."
        )

    return pd.read_csv(target)


def _validate_run_config(config: ElasticityRunConfig) -> None:
    if config.min_samples <= 1:
        raise ValueError("min_samples must be greater than 1")
    if config.min_non_null_pairs is not None and config.min_non_null_pairs <= 1:
        raise ValueError("min_non_null_pairs must be greater than 1 when provided")
    if config.min_unique_price_values <= 1:
        raise ValueError("min_unique_price_values must be greater than 1")
    if config.min_log_price_std <= 0:
        raise ValueError("min_log_price_std must be positive")
    if config.min_log_units_std <= 0:
        raise ValueError("min_log_units_std must be positive")
    if config.epsilon <= 0:
        raise ValueError("epsilon must be strictly positive")
    if config.alpha <= 0 or config.alpha >= 1:
        raise ValueError("alpha must be between 0 and 1")

    validate_causal_feature_profile(config.feature_profile)
    resolve_segment_columns(config.segment_level)
    make_nuisance_models(config.nuisance_model, random_state=config.random_state)


def _estimate_segment_elasticity(model: Any, x_values: np.ndarray) -> float:
    effects = np.asarray(model.const_marginal_effect(x_values)).reshape(-1)
    return float(np.nanmean(effects))


def _estimate_segment_confidence_interval(
    model: Any,
    x_values: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    try:
        lower, upper = model.const_marginal_effect_interval(x_values, alpha=alpha)
        lower_values = np.asarray(lower).reshape(-1)
        upper_values = np.asarray(upper).reshape(-1)
        return float(np.nanmean(lower_values)), float(np.nanmean(upper_values))
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan")


def _normalize_warning_messages(captured_warnings: list[Any]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in captured_warnings:
        message = str(getattr(item, "message", "")).strip()
        if not message or message in seen:
            continue
        normalized.append(message)
        seen.add(message)
    return normalized


def _is_inference_warning(message: str) -> bool:
    lowered = message.strip().lower()
    return any(token in lowered for token in _INFERENCE_WARNING_TOKENS)
