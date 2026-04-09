"""Leakage-safe manual feature engineering pipeline for retail demand forecasting."""

from pathlib import Path
import logging

import pandas as pd

from retail_forecasting.features_common import (
    CALENDAR_FEATURE_COLUMNS,
    DEMAND_LAG_FEATURE_COLUMNS,
    DEMAND_LAG_PERIODS,
    DEMAND_ROLLING_FEATURE_COLUMNS,
    PRICE_FEATURE_COLUMNS,
    add_group_lag_features,
    add_group_shifted_rolling_features,
    clean_binary_series,
    coerce_numeric_columns,
    ensure_group_date_sort_order,
    ordered_unique,
    safe_ratio,
    trim_warmup_rows,
    validate_feature_input_columns,
)
from retail_forecasting.paths import ProjectPaths, build_project_paths
from retail_forecasting.preprocessing import load_json, write_json, write_parquet
from retail_forecasting.schemas import (
    DATE_COLUMN,
    GROUP_COLUMNS,
    PRICE_COLUMN,
    PRODUCT_COLUMN,
    STORE_COLUMN,
    UNITS_COLUMN,
)

LOGGER = logging.getLogger(__name__)

_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "discount",
    "demand_forecast",
    "inventory_level",
    "competitor_price",
    "promotion",
    "holiday",
    "weather",
)

_WARMUP_REQUIRED_COLUMNS: tuple[str, ...] = (
    "units_sold_lag_28",
    "units_sold_roll_mean_28",
    "units_sold_roll_std_28",
    "price_lag_7",
    "price_roll_mean_7",
    "price_roll_std_7",
    "price_momentum_7",
    "price_momentum_28",
)

_LEAKAGE_SAFETY_NOTE = (
    "All target and price rolling features are built from group-wise shifted histories "
    "(shift=1) so each row uses only prior observations."
)

_FORECAST_HINT_NOTE = (
    "forecast_error_hint is intentionally not generated because same-day exogeneity "
    "of demand_forecast cannot be guaranteed from metadata alone."
)


def _resolve_input_path(input_path: Path | None, project_paths: ProjectPaths) -> Path:
    """Resolve input path against project root with a deterministic default."""
    default_path = project_paths.data_interim_dir / "cleaned_retail.parquet"
    candidate = default_path if input_path is None else Path(input_path).expanduser()
    if not candidate.is_absolute():
        candidate = project_paths.project_root / candidate
    return candidate


def _resolve_output_path(output_path: Path | None, project_paths: ProjectPaths) -> Path:
    """Resolve output path against project root with a deterministic default."""
    default_path = project_paths.data_processed_dir / "features_manual.parquet"
    candidate = default_path if output_path is None else Path(output_path).expanduser()
    if not candidate.is_absolute():
        candidate = project_paths.project_root / candidate
    return candidate


def load_cleaned_retail_data(
    input_path: Path | None = None,
    project_paths: ProjectPaths | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Load the cleaned retail parquet used for manual feature generation.

    Args:
        input_path: Optional explicit parquet path.
        project_paths: Optional pre-built repository paths.

    Returns:
        Tuple of resolved parquet path and loaded dataframe.

    Raises:
        FileNotFoundError: If the parquet file does not exist.
    """
    paths = project_paths if project_paths is not None else build_project_paths()
    resolved_path = _resolve_input_path(input_path, paths)

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Cleaned input parquet was not found: {resolved_path}. Run 'prepare-data' first."
        )

    frame = pd.read_parquet(resolved_path)
    LOGGER.info(
        "Loaded cleaned retail data from %s with %d rows and %d columns",
        resolved_path,
        frame.shape[0],
        frame.shape[1],
    )
    return resolved_path, frame


def _prepare_base_frame(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare and normalize base columns before feature generation."""
    required_columns = [DATE_COLUMN, STORE_COLUMN, PRODUCT_COLUMN, UNITS_COLUMN, PRICE_COLUMN]
    validate_feature_input_columns(cleaned_df, required_columns)

    frame = ensure_group_date_sort_order(cleaned_df)
    frame[STORE_COLUMN] = frame[STORE_COLUMN].astype("string").str.strip()
    frame[PRODUCT_COLUMN] = frame[PRODUCT_COLUMN].astype("string").str.strip()
    coerce_numeric_columns(frame, [UNITS_COLUMN, PRICE_COLUMN])
    return frame


def add_calendar_features(dataframe: pd.DataFrame) -> list[str]:
    """Add deterministic calendar features."""
    dataframe["day_of_week"] = dataframe[DATE_COLUMN].dt.dayofweek.astype("int8")
    dataframe["day_of_month"] = dataframe[DATE_COLUMN].dt.day.astype("int8")
    dataframe["day_of_year"] = dataframe[DATE_COLUMN].dt.dayofyear.astype("int16")
    dataframe["week_of_year"] = dataframe[DATE_COLUMN].dt.isocalendar().week.astype("int16")
    dataframe["month"] = dataframe[DATE_COLUMN].dt.month.astype("int8")
    dataframe["quarter"] = dataframe[DATE_COLUMN].dt.quarter.astype("int8")
    dataframe["is_weekend"] = (dataframe["day_of_week"] >= 5).astype("int8")
    dataframe["is_month_start"] = dataframe[DATE_COLUMN].dt.is_month_start.astype("int8")
    dataframe["is_month_end"] = dataframe[DATE_COLUMN].dt.is_month_end.astype("int8")
    return list(CALENDAR_FEATURE_COLUMNS)


def add_demand_features(dataframe: pd.DataFrame) -> list[str]:
    """Add leakage-safe demand lag and rolling features."""
    generated = add_group_lag_features(
        dataframe=dataframe,
        source_column=UNITS_COLUMN,
        lags=DEMAND_LAG_PERIODS,
        feature_prefix=UNITS_COLUMN,
        group_columns=GROUP_COLUMNS,
    )

    generated.extend(
        add_group_shifted_rolling_features(
            dataframe=dataframe,
            source_column=UNITS_COLUMN,
            window_to_statistics={
                7: ("mean", "std", "min", "max"),
                14: ("mean", "std"),
                28: ("mean", "std"),
            },
            feature_prefix=UNITS_COLUMN,
            group_columns=GROUP_COLUMNS,
            shift_periods=1,
        )
    )

    return generated


def add_price_features(dataframe: pd.DataFrame) -> list[str]:
    """Add leakage-safe price lag, change, and momentum features."""
    generated = add_group_lag_features(
        dataframe=dataframe,
        source_column=PRICE_COLUMN,
        lags=(1, 7),
        feature_prefix=PRICE_COLUMN,
        group_columns=GROUP_COLUMNS,
    )

    dataframe["price_change_1d"] = dataframe[PRICE_COLUMN] - dataframe["price_lag_1"]
    dataframe["price_change_7d"] = dataframe[PRICE_COLUMN] - dataframe["price_lag_7"]
    generated.extend(["price_change_1d", "price_change_7d"])

    generated.extend(
        add_group_shifted_rolling_features(
            dataframe=dataframe,
            source_column=PRICE_COLUMN,
            window_to_statistics={7: ("mean", "std")},
            feature_prefix=PRICE_COLUMN,
            group_columns=GROUP_COLUMNS,
            shift_periods=1,
        )
    )

    grouped_price = dataframe.groupby(list(GROUP_COLUMNS), sort=False)[PRICE_COLUMN]
    prior_price_roll_mean_28 = grouped_price.transform(
        lambda series: series.shift(1).rolling(window=28, min_periods=28).mean()
    )

    dataframe["price_momentum_7"] = safe_ratio(dataframe[PRICE_COLUMN], dataframe["price_roll_mean_7"])
    dataframe["price_momentum_28"] = safe_ratio(dataframe[PRICE_COLUMN], prior_price_roll_mean_28)
    generated.extend(["price_momentum_7", "price_momentum_28"])

    return generated


def _add_discount_and_forecast_features(
    dataframe: pd.DataFrame,
    optional_groups_used: list[str],
    missing_columns: list[str],
) -> list[str]:
    """Add optional discount and demand forecast passthrough features."""
    generated: list[str] = []

    if "discount" in dataframe.columns:
        coerce_numeric_columns(dataframe, ["discount"])
        generated.append("discount")
        optional_groups_used.append("discount")
    else:
        missing_columns.append("discount")

    if "demand_forecast" in dataframe.columns:
        coerce_numeric_columns(dataframe, ["demand_forecast"])
        generated.append("demand_forecast")
        optional_groups_used.append("demand_forecast")
    else:
        missing_columns.append("demand_forecast")

    return generated


def _add_inventory_features(
    dataframe: pd.DataFrame,
    optional_groups_used: list[str],
    missing_columns: list[str],
) -> list[str]:
    """Add optional inventory passthrough and conservative stock-related flags."""
    if "inventory_level" not in dataframe.columns:
        missing_columns.append("inventory_level")
        return []

    coerce_numeric_columns(dataframe, ["inventory_level"])
    grouped_inventory = dataframe.groupby(list(GROUP_COLUMNS), sort=False)["inventory_level"]
    prior_inventory_median = grouped_inventory.transform(
        lambda series: series.shift(1).rolling(window=28, min_periods=7).median()
    )

    fallback_inventory = float(dataframe["inventory_level"].median())
    threshold = (prior_inventory_median.fillna(fallback_inventory) * 0.5).clip(lower=0.0)

    dataframe["low_inventory_flag"] = (dataframe["inventory_level"] <= threshold).astype("int8")
    dataframe["stockout_proxy"] = (
        (dataframe["inventory_level"] <= 0) & (dataframe[UNITS_COLUMN] <= 0)
    ).astype("int8")

    optional_groups_used.append("inventory")
    return ["inventory_level", "low_inventory_flag", "stockout_proxy"]


def _add_exogenous_features(
    dataframe: pd.DataFrame,
    optional_groups_used: list[str],
    missing_columns: list[str],
) -> list[str]:
    """Add optional exogenous feature transformations."""
    generated: list[str] = []

    if "competitor_price" in dataframe.columns:
        coerce_numeric_columns(dataframe, ["competitor_price"])
        dataframe["price_vs_competitor_abs"] = dataframe[PRICE_COLUMN] - dataframe["competitor_price"]
        dataframe["price_vs_competitor_pct"] = safe_ratio(
            dataframe["price_vs_competitor_abs"], dataframe["competitor_price"]
        )
        generated.extend(["competitor_price", "price_vs_competitor_abs", "price_vs_competitor_pct"])
        optional_groups_used.append("competitor_price")
    else:
        missing_columns.append("competitor_price")

    if "promotion" in dataframe.columns:
        dataframe["promotion"] = clean_binary_series(dataframe["promotion"])
        generated.append("promotion")
        optional_groups_used.append("promotion")
    else:
        missing_columns.append("promotion")

    if "holiday" in dataframe.columns:
        dataframe["holiday"] = clean_binary_series(dataframe["holiday"])
        generated.append("holiday")
        optional_groups_used.append("holiday")
    else:
        missing_columns.append("holiday")

    if "weather" in dataframe.columns:
        normalized_weather = dataframe["weather"].astype("string").str.strip().str.lower().fillna("missing")
        categories = sorted(set(normalized_weather.tolist()))
        weather_mapping = {value: index for index, value in enumerate(categories)}
        dataframe["weather_code"] = normalized_weather.map(weather_mapping).astype("float64")
        generated.append("weather_code")
        optional_groups_used.append("weather")
    else:
        missing_columns.append("weather")

    return generated


def generate_manual_features_frame(
    cleaned_df: pd.DataFrame,
    drop_warmup_rows: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Generate leakage-safe manual features from cleaned retail data.

    Args:
        cleaned_df: Cleaned, canonical retail dataframe.
        drop_warmup_rows: Whether to remove rows without required lag/rolling history.

    Returns:
        Tuple of engineered dataframe and metadata summary.
    """
    frame = _prepare_base_frame(cleaned_df)
    original_row_count = int(frame.shape[0])

    optional_groups_used: list[str] = []
    missing_optional_columns: list[str] = []

    generated_features: list[str] = []
    generated_features.extend(add_calendar_features(frame))
    generated_features.extend(add_demand_features(frame))
    generated_features.extend(add_price_features(frame))
    generated_features.extend(
        _add_discount_and_forecast_features(frame, optional_groups_used, missing_optional_columns)
    )
    generated_features.extend(_add_inventory_features(frame, optional_groups_used, missing_optional_columns))
    generated_features.extend(_add_exogenous_features(frame, optional_groups_used, missing_optional_columns))

    generated_features = ordered_unique(generated_features)
    missing_optional_columns = ordered_unique(missing_optional_columns)
    optional_groups_used = ordered_unique(optional_groups_used)

    base_columns = [DATE_COLUMN, STORE_COLUMN, PRODUCT_COLUMN, UNITS_COLUMN, PRICE_COLUMN]
    output_columns = ordered_unique(base_columns + generated_features)
    feature_frame = frame[output_columns].copy(deep=True)

    dropped_for_warmup = 0
    if drop_warmup_rows:
        feature_frame, dropped_for_warmup = trim_warmup_rows(feature_frame, _WARMUP_REQUIRED_COLUMNS)

    if feature_frame.empty:
        raise ValueError(
            "Feature generation produced an empty dataframe. "
            "This usually means there is not enough history for 28-day lag/rolling features."
        )

    metadata = {
        "row_count_before_feature_generation": original_row_count,
        "row_count_after_warmup_trimming": int(feature_frame.shape[0]),
        "rows_dropped_for_warmup": int(dropped_for_warmup),
        "feature_column_count": int(len(generated_features)),
        "generated_features": generated_features,
        "optional_feature_groups_used": optional_groups_used,
        "columns_skipped_missing": missing_optional_columns,
        "has_missing_optional_columns": bool(missing_optional_columns),
        "warmup_required_columns": list(_WARMUP_REQUIRED_COLUMNS),
        "forecast_error_hint_generated": False,
    }

    return feature_frame, metadata


def _write_split_feature_artifacts(
    feature_frame: pd.DataFrame,
    project_paths: ProjectPaths,
) -> tuple[dict[str, Path], dict[str, int]]:
    """Write split-specific feature artifacts when split files are available."""
    split_source_paths: dict[str, Path] = {
        "train": project_paths.data_processed_dir / "train.parquet",
        "validation": project_paths.data_processed_dir / "validation.parquet",
        "test": project_paths.data_processed_dir / "test.parquet",
    }

    split_output_paths: dict[str, Path] = {}
    split_row_counts: dict[str, int] = {}

    feature_dates = pd.to_datetime(feature_frame[DATE_COLUMN], errors="coerce").dt.floor("D")

    for split_name, source_path in split_source_paths.items():
        if not source_path.exists():
            continue

        split_frame = pd.read_parquet(source_path)
        if DATE_COLUMN not in split_frame.columns:
            continue

        split_dates = pd.to_datetime(split_frame[DATE_COLUMN], errors="coerce").dropna().dt.floor("D")
        split_feature_frame = feature_frame.loc[feature_dates.isin(set(split_dates))].reset_index(drop=True)

        output_path = project_paths.data_processed_dir / f"features_{split_name}.parquet"
        split_output_paths[f"features_{split_name}_parquet"] = write_parquet(split_feature_frame, output_path)
        split_row_counts[f"{split_name}_row_count"] = int(split_feature_frame.shape[0])

    return split_output_paths, split_row_counts


def build_manual_features_pipeline(
    input_path: Path | None = None,
    output_path: Path | None = None,
    drop_warmup_rows: bool = True,
    write_split_artifacts: bool = True,
) -> dict[str, Path]:
    """Build manual features and persist artifacts for downstream modeling.

    Args:
        input_path: Optional explicit cleaned parquet path.
        output_path: Optional explicit features parquet path.
        drop_warmup_rows: Whether to trim warmup rows.
        write_split_artifacts: Whether to write split-specific feature outputs.

    Returns:
        Mapping of produced artifact names to output paths.
    """
    project_paths = build_project_paths()
    source_path, cleaned_df = load_cleaned_retail_data(input_path=input_path, project_paths=project_paths)
    feature_frame, metadata = generate_manual_features_frame(
        cleaned_df,
        drop_warmup_rows=drop_warmup_rows,
    )

    resolved_output_path = _resolve_output_path(output_path, project_paths)
    output_paths: dict[str, Path] = {
        "features_manual_parquet": write_parquet(feature_frame, resolved_output_path)
    }

    split_row_counts: dict[str, int] = {}
    if write_split_artifacts:
        split_paths, split_row_counts = _write_split_feature_artifacts(feature_frame, project_paths)
        output_paths.update(split_paths)

    summary_payload: dict[str, object] = {
        "source_input_path": str(source_path),
        "output_paths": {name: str(path) for name, path in output_paths.items()},
        "leakage_safety_note": _LEAKAGE_SAFETY_NOTE,
        "forecast_error_hint_note": _FORECAST_HINT_NOTE,
        **metadata,
    }

    if split_row_counts:
        summary_payload["split_row_counts"] = split_row_counts

    summary_path = write_json(summary_payload, project_paths.artifacts_dir / "features_manual_summary.json")
    output_paths["features_manual_summary_json"] = summary_path

    LOGGER.info(
        "Generated manual features with %d rows and %d columns",
        feature_frame.shape[0],
        feature_frame.shape[1],
    )
    return output_paths


def load_features_summary(summary_path: Path | None = None) -> dict[str, object]:
    """Load the feature summary JSON artifact.

    Args:
        summary_path: Optional explicit summary path.

    Returns:
        Parsed summary payload.
    """
    paths = build_project_paths()
    target_path = paths.artifacts_dir / "features_manual_summary.json" if summary_path is None else Path(summary_path)
    if not target_path.is_absolute():
        target_path = paths.project_root / target_path

    if not target_path.exists():
        raise FileNotFoundError(
            f"Missing feature summary artifact: {target_path}. Run 'build-manual-features' first."
        )

    payload = load_json(target_path)
    return payload


def expected_manual_feature_names() -> tuple[str, ...]:
    """Return stable expected core manual feature names for validation and tests."""
    return (
        *CALENDAR_FEATURE_COLUMNS,
        *DEMAND_LAG_FEATURE_COLUMNS,
        *DEMAND_ROLLING_FEATURE_COLUMNS,
        *PRICE_FEATURE_COLUMNS,
    )
