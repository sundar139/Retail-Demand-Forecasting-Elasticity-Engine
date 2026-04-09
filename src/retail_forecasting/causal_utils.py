"""Utilities for causal price-elasticity estimation with explicit safety guardrails."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import logging

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from retail_forecasting.features_common import ensure_group_date_sort_order
from retail_forecasting.paths import ProjectPaths, build_project_paths
from retail_forecasting.preprocessing import load_json
from retail_forecasting.schemas import DATE_COLUMN, PRICE_COLUMN, PRODUCT_COLUMN, STORE_COLUMN, UNITS_COLUMN

LOGGER = logging.getLogger(__name__)

SEGMENT_LEVEL_PRODUCT = "product"
SEGMENT_LEVEL_STORE_PRODUCT = "store-product"
ALLOWED_SEGMENT_LEVELS: frozenset[str] = frozenset({SEGMENT_LEVEL_PRODUCT, SEGMENT_LEVEL_STORE_PRODUCT})

CAUSAL_FEATURE_PROFILE_LEAN = "lean"
CAUSAL_FEATURE_PROFILE_FULL = "full"
ALLOWED_CAUSAL_FEATURE_PROFILES: frozenset[str] = frozenset(
    {CAUSAL_FEATURE_PROFILE_LEAN, CAUSAL_FEATURE_PROFILE_FULL}
)

DEMAND_FORECAST_COLUMN = "demand_forecast"
LEAN_CAUSAL_CONTROL_COLUMNS: tuple[str, ...] = (
    "day_of_week",
    "month",
    "is_weekend",
    "units_sold_lag_7",
    "units_sold_lag_28",
    "inventory_level",
    "discount",
    "competitor_price",
    "promotion",
    "holiday",
)

IDENTIFIER_COLUMNS: frozenset[str] = frozenset({DATE_COLUMN, STORE_COLUMN, PRODUCT_COLUMN})

FORBIDDEN_CONTROL_COLUMNS: frozenset[str] = frozenset(
    {
        UNITS_COLUMN,
        PRICE_COLUMN,
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
    }
)


@dataclass(frozen=True, slots=True)
class LoadedCausalData:
    """Container for loaded causal inputs and augmentation metadata."""

    source_data_path: Path
    frame: pd.DataFrame
    llm_augmentation_used: bool
    llm_added_columns: list[str]
    notes: list[str]


@dataclass(frozen=True, slots=True)
class SegmentGuardrailResult:
    """Validation summary for one segment before causal model fitting."""

    is_valid: bool
    reason: str | None
    sample_size: int
    non_null_pair_count: int
    unique_treatment_values: int
    treatment_std: float
    outcome_std: float
    treatment_variance: float
    outcome_variance: float


def safe_log_transform(series: pd.Series, epsilon: float) -> pd.Series:
    """Apply numerically safe log transform to a numeric series.

    Args:
        series: Numeric series to transform.
        epsilon: Positive additive constant.

    Returns:
        Log-transformed float series.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be strictly positive")

    values = pd.to_numeric(series, errors="coerce").astype("float64")
    invalid_mask = values < -epsilon
    if invalid_mask.any():
        count = int(invalid_mask.sum())
        raise ValueError(
            f"safe_log_transform received {count} values < -epsilon, which would make log undefined"
        )

    logged_values = np.log(values.to_numpy(dtype="float64") + float(epsilon))
    return pd.Series(logged_values, index=values.index, name=series.name, dtype="float64")


def resolve_segment_columns(segment_level: str) -> tuple[str, ...]:
    """Resolve segmenting columns for supported segment levels."""
    normalized = segment_level.strip().lower()
    if normalized == SEGMENT_LEVEL_PRODUCT:
        return (PRODUCT_COLUMN,)
    if normalized == SEGMENT_LEVEL_STORE_PRODUCT:
        return (STORE_COLUMN, PRODUCT_COLUMN)

    allowed = ", ".join(sorted(ALLOWED_SEGMENT_LEVELS))
    raise ValueError(f"Unsupported segment_level '{segment_level}'. Allowed values: {allowed}")


def validate_causal_feature_profile(feature_profile: str) -> str:
    """Validate and normalize causal control feature profile name."""
    normalized = feature_profile.strip().lower()
    if normalized not in ALLOWED_CAUSAL_FEATURE_PROFILES:
        allowed = ", ".join(sorted(ALLOWED_CAUSAL_FEATURE_PROFILES))
        raise ValueError(
            f"Unsupported feature_profile '{feature_profile}'. Allowed values: {allowed}"
        )
    return normalized


def format_segment_key(segment_columns: tuple[str, ...], segment_values: object) -> str:
    """Build a deterministic segment-key string for reporting artifacts."""
    if not isinstance(segment_values, tuple):
        segment_tuple = (segment_values,)
    else:
        segment_tuple = segment_values

    pairs = [
        f"{column_name}={value}"
        for column_name, value in zip(segment_columns, segment_tuple, strict=True)
    ]
    return "|".join(pairs)


def add_log_outcome_treatment(
    frame: pd.DataFrame,
    epsilon: float,
    outcome_column: str = UNITS_COLUMN,
    treatment_column: str = PRICE_COLUMN,
) -> pd.DataFrame:
    """Add log-transformed outcome and treatment columns to a frame copy."""
    transformed = frame.copy(deep=True)
    transformed["y_log"] = safe_log_transform(transformed[outcome_column], epsilon=epsilon)
    transformed["t_log"] = safe_log_transform(transformed[treatment_column], epsilon=epsilon)
    return transformed


def evaluate_segment_guardrails(
    segment_frame: pd.DataFrame,
    outcome_column: str,
    treatment_column: str,
    min_samples: int,
    min_non_null_pairs: int | None = None,
    min_unique_treatment_values: int = 8,
    min_treatment_std: float = 1e-3,
    min_outcome_std: float = 1e-3,
    min_variance: float = 1e-8,
) -> SegmentGuardrailResult:
    """Evaluate pre-fit quality checks for one segment."""
    sample_size = int(len(segment_frame))
    resolved_min_non_null_pairs = min_samples if min_non_null_pairs is None else min_non_null_pairs

    if min_samples <= 1:
        raise ValueError("min_samples must be greater than 1")
    if resolved_min_non_null_pairs <= 1:
        raise ValueError("min_non_null_pairs must be greater than 1")
    if min_unique_treatment_values <= 1:
        raise ValueError("min_unique_treatment_values must be greater than 1")
    if min_treatment_std <= 0:
        raise ValueError("min_treatment_std must be positive")
    if min_outcome_std <= 0:
        raise ValueError("min_outcome_std must be positive")

    if sample_size < min_samples:
        return SegmentGuardrailResult(
            is_valid=False,
            reason=f"sample_size_below_threshold ({sample_size} < {min_samples})",
            sample_size=sample_size,
            non_null_pair_count=0,
            unique_treatment_values=0,
            treatment_std=0.0,
            outcome_std=0.0,
            treatment_variance=0.0,
            outcome_variance=0.0,
        )

    outcome_series = pd.to_numeric(segment_frame[outcome_column], errors="coerce").astype("float64")
    treatment_series = pd.to_numeric(segment_frame[treatment_column], errors="coerce").astype("float64")
    complete_mask = outcome_series.notna() & treatment_series.notna()

    complete_count = int(complete_mask.sum())
    if complete_count < resolved_min_non_null_pairs:
        return SegmentGuardrailResult(
            is_valid=False,
            reason=(
                f"non_null_pair_count_below_threshold ({complete_count} < {resolved_min_non_null_pairs}) "
                f"for {outcome_column}/{treatment_column}"
            ),
            sample_size=sample_size,
            non_null_pair_count=complete_count,
            unique_treatment_values=0,
            treatment_std=0.0,
            outcome_std=0.0,
            treatment_variance=0.0,
            outcome_variance=0.0,
        )

    outcome_valid = outcome_series.loc[complete_mask]
    treatment_valid = treatment_series.loc[complete_mask]

    outcome_variance = float(outcome_valid.var(ddof=0))
    treatment_variance = float(treatment_valid.var(ddof=0))
    outcome_std = float(outcome_valid.std(ddof=0))
    treatment_std = float(treatment_valid.std(ddof=0))
    unique_treatment_values = int(treatment_valid.nunique(dropna=True))

    if unique_treatment_values < min_unique_treatment_values:
        return SegmentGuardrailResult(
            is_valid=False,
            reason=(
                "unique_price_values_below_threshold "
                f"({unique_treatment_values} < {min_unique_treatment_values})"
            ),
            sample_size=sample_size,
            non_null_pair_count=complete_count,
            unique_treatment_values=unique_treatment_values,
            treatment_std=treatment_std,
            outcome_std=outcome_std,
            treatment_variance=treatment_variance,
            outcome_variance=outcome_variance,
        )
    if float(outcome_valid.nunique(dropna=True)) < 2:
        return SegmentGuardrailResult(
            is_valid=False,
            reason="insufficient_outcome_variation",
            sample_size=sample_size,
            non_null_pair_count=complete_count,
            unique_treatment_values=unique_treatment_values,
            treatment_std=treatment_std,
            outcome_std=outcome_std,
            treatment_variance=treatment_variance,
            outcome_variance=outcome_variance,
        )

    if treatment_std < min_treatment_std:
        return SegmentGuardrailResult(
            is_valid=False,
            reason=(
                "log_price_std_below_threshold "
                f"({treatment_std:.6f} < {min_treatment_std:.6f})"
            ),
            sample_size=sample_size,
            non_null_pair_count=complete_count,
            unique_treatment_values=unique_treatment_values,
            treatment_std=treatment_std,
            outcome_std=outcome_std,
            treatment_variance=treatment_variance,
            outcome_variance=outcome_variance,
        )

    if outcome_std < min_outcome_std:
        return SegmentGuardrailResult(
            is_valid=False,
            reason=(
                "log_units_std_below_threshold "
                f"({outcome_std:.6f} < {min_outcome_std:.6f})"
            ),
            sample_size=sample_size,
            non_null_pair_count=complete_count,
            unique_treatment_values=unique_treatment_values,
            treatment_std=treatment_std,
            outcome_std=outcome_std,
            treatment_variance=treatment_variance,
            outcome_variance=outcome_variance,
        )

    if treatment_variance <= min_variance:
        return SegmentGuardrailResult(
            is_valid=False,
            reason=f"treatment_variance_too_low ({treatment_variance:.3e})",
            sample_size=sample_size,
            non_null_pair_count=complete_count,
            unique_treatment_values=unique_treatment_values,
            treatment_std=treatment_std,
            outcome_std=outcome_std,
            treatment_variance=treatment_variance,
            outcome_variance=outcome_variance,
        )
    if outcome_variance <= min_variance:
        return SegmentGuardrailResult(
            is_valid=False,
            reason=f"outcome_variance_too_low ({outcome_variance:.3e})",
            sample_size=sample_size,
            non_null_pair_count=complete_count,
            unique_treatment_values=unique_treatment_values,
            treatment_std=treatment_std,
            outcome_std=outcome_std,
            treatment_variance=treatment_variance,
            outcome_variance=outcome_variance,
        )

    return SegmentGuardrailResult(
        is_valid=True,
        reason=None,
        sample_size=sample_size,
        non_null_pair_count=complete_count,
        unique_treatment_values=unique_treatment_values,
        treatment_std=treatment_std,
        outcome_std=outcome_std,
        treatment_variance=treatment_variance,
        outcome_variance=outcome_variance,
    )


def load_causal_feature_frame(
    input_path: Path | None = None,
    use_llm_features: bool = True,
    llm_features_path: Path | None = None,
    project_paths: ProjectPaths | None = None,
) -> LoadedCausalData:
    """Load primary manual features and optionally augment with valid LLM-derived features."""
    paths = project_paths if project_paths is not None else build_project_paths()

    manual_source = paths.data_processed_dir / "features_manual.parquet"
    source_path = Path(input_path).expanduser() if input_path is not None else manual_source
    if not source_path.is_absolute():
        source_path = paths.project_root / source_path

    if not source_path.exists():
        raise FileNotFoundError(
            f"Primary causal input not found: {source_path}. Run 'build-manual-features' first."
        )

    frame = pd.read_parquet(source_path)
    frame = ensure_group_date_sort_order(frame)

    required = {DATE_COLUMN, STORE_COLUMN, PRODUCT_COLUMN, UNITS_COLUMN, PRICE_COLUMN}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Primary causal input is missing required columns: {', '.join(missing)}"
        )

    notes: list[str] = []
    llm_added_columns: list[str] = []
    llm_used = False

    if use_llm_features:
        llm_source = (
            paths.data_processed_dir / "features_llm.parquet"
            if llm_features_path is None
            else Path(llm_features_path).expanduser()
        )
        if not llm_source.is_absolute():
            llm_source = paths.project_root / llm_source

        if not llm_source.exists():
            notes.append(f"LLM feature augmentation skipped: file not found at {llm_source}")
        else:
            llm_frame = pd.read_parquet(llm_source)
            if llm_frame.empty:
                notes.append("LLM feature augmentation skipped: LLM feature file is empty")
            elif not {DATE_COLUMN, STORE_COLUMN, PRODUCT_COLUMN}.issubset(llm_frame.columns):
                notes.append("LLM feature augmentation skipped: missing merge keys in LLM feature file")
            else:
                validated_names, validation_notes = _load_validated_llm_feature_names(paths)
                notes.extend(validation_notes)
                if validated_names is not None and not validated_names:
                    return LoadedCausalData(
                        source_data_path=source_path,
                        frame=frame,
                        llm_augmentation_used=False,
                        llm_added_columns=[],
                        notes=notes,
                    )

                llm_frame = ensure_group_date_sort_order(llm_frame)
                merge_keys = [DATE_COLUMN, STORE_COLUMN, PRODUCT_COLUMN]
                llm_candidate_columns = [
                    column_name
                    for column_name in llm_frame.columns
                    if column_name not in merge_keys and column_name not in frame.columns
                ]

                if validated_names is not None:
                    allowed_names = set(validated_names)
                    llm_candidate_columns = [
                        column_name
                        for column_name in llm_candidate_columns
                        if column_name in allowed_names
                    ]

                if not llm_candidate_columns:
                    notes.append(
                        "LLM feature augmentation skipped: no non-overlapping feature columns found"
                    )
                else:
                    llm_subset = llm_frame[merge_keys + llm_candidate_columns].drop_duplicates(
                        subset=merge_keys,
                        keep="last",
                    )

                    ordered = frame.reset_index(drop=False).rename(columns={"index": "_row_order"})
                    merged = ordered.merge(
                        llm_subset,
                        on=merge_keys,
                        how="left",
                        sort=False,
                        validate="m:1",
                    )
                    frame = (
                        merged.sort_values("_row_order", kind="mergesort")
                        .drop(columns=["_row_order"])
                        .reset_index(drop=True)
                    )
                    frame = ensure_group_date_sort_order(frame)
                    llm_added_columns = llm_candidate_columns
                    llm_used = bool(llm_added_columns)
                    notes.append(
                        f"LLM feature augmentation used with {len(llm_added_columns)} columns"
                    )

    return LoadedCausalData(
        source_data_path=source_path,
        frame=frame,
        llm_augmentation_used=llm_used,
        llm_added_columns=llm_added_columns,
        notes=notes,
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


def select_causal_control_features(
    frame: pd.DataFrame,
    feature_profile: str = CAUSAL_FEATURE_PROFILE_LEAN,
    treatment_column: str = PRICE_COLUMN,
    include_store_indicators: bool = True,
    include_llm_derived: bool = False,
    allow_demand_forecast: bool = False,
    extra_excluded_columns: Sequence[str] | None = None,
) -> list[str]:
    """Select causal control features under explicit lean/full profiles.

    The lean profile is intentionally compact and auditable for own-price elasticity.
    """
    profile = validate_causal_feature_profile(feature_profile)

    excluded = set(FORBIDDEN_CONTROL_COLUMNS)
    excluded.update(IDENTIFIER_COLUMNS)
    excluded.update(str(name) for name in (extra_excluded_columns or []))

    selected: list[str] = []
    allowed_lean = set(LEAN_CAUSAL_CONTROL_COLUMNS)

    for column_name in frame.columns:
        if column_name in excluded:
            continue

        lowered = column_name.lower()
        if lowered.startswith("target_"):
            continue
        if "future" in lowered:
            continue
        if _is_own_price_proxy_feature(column_name, treatment_column=treatment_column):
            continue
        if not include_llm_derived and _is_llm_derived_feature(column_name):
            continue
        if lowered == DEMAND_FORECAST_COLUMN and not allow_demand_forecast:
            continue

        if profile == CAUSAL_FEATURE_PROFILE_LEAN:
            is_store_indicator = include_store_indicators and _is_store_indicator_feature(column_name)
            if column_name not in allowed_lean and not is_store_indicator:
                continue

        is_numeric = pd.api.types.is_numeric_dtype(frame[column_name])
        is_boolean = pd.api.types.is_bool_dtype(frame[column_name])
        if not is_numeric and not is_boolean:
            continue

        selected.append(str(column_name))

    return selected


def select_control_feature_columns(
    frame: pd.DataFrame,
    extra_excluded_columns: Sequence[str] | None = None,
) -> list[str]:
    """Backward-compatible control selector using permissive full profile semantics."""
    return select_causal_control_features(
        frame,
        feature_profile=CAUSAL_FEATURE_PROFILE_FULL,
        include_store_indicators=True,
        include_llm_derived=True,
        allow_demand_forecast=True,
        extra_excluded_columns=extra_excluded_columns,
    )


def _is_own_price_proxy_feature(column_name: str, treatment_column: str) -> bool:
    lowered = column_name.strip().lower()
    treatment_token = treatment_column.strip().lower()
    if not treatment_token or treatment_token not in lowered:
        return False

    proxy_tokens = (
        "_lag_",
        "_roll_",
        "_momentum",
        "_delta",
        "_change",
        "_ma_",
        "_ewm",
    )
    return any(token in lowered for token in proxy_tokens)


def _is_llm_derived_feature(column_name: str) -> bool:
    lowered = column_name.strip().lower()
    return lowered.startswith("llm_") or lowered.endswith("_llm") or "_llm_" in lowered


def _is_store_indicator_feature(column_name: str) -> bool:
    lowered = column_name.strip().lower()
    if lowered in {"store_id_encoded", "store_numeric_id", "store_fe", "store_fixed_effect"}:
        return True
    return lowered.startswith("store_indicator_")


def prepare_segment_controls(
    segment_frame: pd.DataFrame,
    control_columns: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare per-segment control matrix with deterministic imputation and filtering."""
    usable_columns: list[str] = []
    prepared = pd.DataFrame(index=segment_frame.index)

    for column_name in control_columns:
        if column_name not in segment_frame.columns:
            continue

        numeric = pd.to_numeric(segment_frame[column_name], errors="coerce").astype("float64")
        if numeric.notna().sum() < 2:
            continue
        if float(numeric.var(ddof=0)) <= 0:
            continue

        median_value = float(numeric.median(skipna=True)) if numeric.notna().any() else 0.0
        if np.isnan(median_value):
            median_value = 0.0
        prepared[column_name] = numeric.fillna(median_value)
        usable_columns.append(column_name)

    return prepared, usable_columns


def make_nuisance_models(
    model_name: str,
    random_state: int = 42,
) -> tuple[object, object, str]:
    """Build default nuisance regressors for DML nuisance estimation."""
    normalized = model_name.strip().lower()

    if normalized == "random-forest":
        model_y = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
        model_t = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
        return model_y, model_t, "random-forest"

    if normalized == "gradient-boosting":
        model_y = GradientBoostingRegressor(random_state=random_state)
        model_t = GradientBoostingRegressor(random_state=random_state)
        return model_y, model_t, "gradient-boosting"

    raise ValueError(
        "Unsupported nuisance model. Allowed values: random-forest, gradient-boosting"
    )


def generate_elasticity_report_artifacts(
    estimates: pd.DataFrame,
    project_paths: ProjectPaths | None = None,
    top_n: int = 10,
    max_ci_segments: int = 20,
) -> dict[str, Path]:
    """Generate histogram, top-segment CSV, and optional CI plot artifacts."""
    paths = project_paths if project_paths is not None else build_project_paths()
    paths.reports_figures_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    successful = estimates.loc[
        (estimates["fit_status"] == "success") & estimates["elasticity_estimate"].notna()
    ].copy()

    if successful.empty:
        LOGGER.info("No successful elasticity estimates available; skipping plot generation")
        return outputs

    histogram_path = paths.reports_figures_dir / "elasticity_estimate_histogram.png"
    figure = plt.figure(figsize=(8, 5))
    ax = figure.add_subplot(1, 1, 1)
    ax.hist(successful["elasticity_estimate"], bins=25, edgecolor="black", alpha=0.8)
    ax.set_title("Elasticity Estimate Distribution")
    ax.set_xlabel("Estimated own-price elasticity")
    ax.set_ylabel("Segment count")
    figure.tight_layout()
    figure.savefig(histogram_path, dpi=120)
    plt.close(figure)
    outputs["elasticity_histogram_png"] = histogram_path

    most_elastic = successful.nsmallest(top_n, columns=["elasticity_estimate"]).copy()
    most_elastic["elasticity_rank_group"] = "most_elastic"

    least_elastic = successful.nlargest(top_n, columns=["elasticity_estimate"]).copy()
    least_elastic["elasticity_rank_group"] = "least_elastic"

    top_segments = pd.concat([most_elastic, least_elastic], axis=0, ignore_index=True)
    top_segments_path = paths.reports_figures_dir / "elasticity_top_segments.csv"
    top_segments.to_csv(top_segments_path, index=False)
    outputs["elasticity_top_segments_csv"] = top_segments_path

    ci_ready = successful.dropna(subset=["lower_ci", "upper_ci"]).copy()
    if not ci_ready.empty:
        ci_subset = ci_ready.nsmallest(max_ci_segments, columns=["elasticity_estimate"]).copy()
        ci_subset = ci_subset.reset_index(drop=True)

        ci_plot_path = paths.reports_figures_dir / "elasticity_ci_segments.png"
        figure = plt.figure(figsize=(10, max(4, 0.35 * len(ci_subset))))
        ax = figure.add_subplot(1, 1, 1)

        y_positions = np.arange(len(ci_subset))
        point_estimates = ci_subset["elasticity_estimate"].to_numpy(dtype=float)
        lower_values = ci_subset["lower_ci"].to_numpy(dtype=float)
        upper_values = ci_subset["upper_ci"].to_numpy(dtype=float)

        lower_error = point_estimates - lower_values
        upper_error = upper_values - point_estimates

        ax.errorbar(
            x=point_estimates,
            y=y_positions,
            xerr=[lower_error, upper_error],
            fmt="o",
            color="#0b4f6c",
            ecolor="#6c757d",
            capsize=3,
        )
        ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(ci_subset["segment_key"].tolist())
        ax.set_title("Elasticity Estimates with 95% CI")
        ax.set_xlabel("Estimated own-price elasticity")
        ax.set_ylabel("Segment")
        figure.tight_layout()
        figure.savefig(ci_plot_path, dpi=120)
        plt.close(figure)
        outputs["elasticity_ci_plot_png"] = ci_plot_path

    return outputs
