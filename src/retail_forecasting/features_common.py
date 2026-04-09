"""Reusable helpers for leakage-safe manual feature engineering."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Final

import numpy as np
import pandas as pd

from retail_forecasting.schemas import DATE_COLUMN, GROUP_COLUMNS

CALENDAR_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "week_of_year",
    "month",
    "quarter",
    "is_weekend",
    "is_month_start",
    "is_month_end",
)

DEMAND_LAG_PERIODS: Final[tuple[int, ...]] = (1, 7, 14, 28)

DEMAND_LAG_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"units_sold_lag_{lag}" for lag in DEMAND_LAG_PERIODS
)

DEMAND_ROLLING_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "units_sold_roll_mean_7",
    "units_sold_roll_std_7",
    "units_sold_roll_min_7",
    "units_sold_roll_max_7",
    "units_sold_roll_mean_14",
    "units_sold_roll_std_14",
    "units_sold_roll_mean_28",
    "units_sold_roll_std_28",
)

PRICE_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "price_lag_1",
    "price_lag_7",
    "price_change_1d",
    "price_change_7d",
    "price_roll_mean_7",
    "price_roll_std_7",
    "price_momentum_7",
    "price_momentum_28",
)


def validate_feature_input_columns(dataframe: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """Validate that required columns are available in a dataframe.

    Args:
        dataframe: Input dataframe.
        required_columns: Required columns.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = sorted(set(required_columns) - set(dataframe.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required columns for feature generation: {missing_text}")


def ensure_group_date_sort_order(
    dataframe: pd.DataFrame,
    group_columns: Sequence[str] = GROUP_COLUMNS,
    date_column: str = DATE_COLUMN,
) -> pd.DataFrame:
    """Sort records by group keys and date using deterministic ordering.

    Args:
        dataframe: Input dataframe.
        group_columns: Grouping columns.
        date_column: Date column name.

    Returns:
        Sorted dataframe copy.
    """
    sorted_frame = dataframe.copy(deep=True)
    sorted_frame[date_column] = pd.to_datetime(sorted_frame[date_column], errors="coerce").dt.floor("D")
    if sorted_frame[date_column].isna().any():
        raise ValueError(f"Column '{date_column}' has invalid date values after parsing")

    sort_columns = list(group_columns) + [date_column]
    return sorted_frame.sort_values(by=sort_columns, kind="mergesort").reset_index(drop=True)


def coerce_numeric_columns(dataframe: pd.DataFrame, columns: Iterable[str]) -> None:
    """Coerce selected columns to numeric dtype in place.

    Args:
        dataframe: Input dataframe.
        columns: Columns to coerce.

    Raises:
        ValueError: If a present column cannot be parsed as numeric.
    """
    for column in columns:
        if column not in dataframe.columns:
            continue
        converted = pd.to_numeric(dataframe[column], errors="coerce")
        if converted.isna().any():
            invalid_count = int(converted.isna().sum())
            raise ValueError(
                f"Column '{column}' contains {invalid_count} non-numeric values that cannot be coerced"
            )
        dataframe[column] = converted.astype("float64")


def clean_binary_series(series: pd.Series) -> pd.Series:
    """Convert common boolean-like strings and values into numeric binary values.

    Args:
        series: Input series.

    Returns:
        Float series containing 1.0, 0.0, or NaN.
    """
    true_values = {"1", "true", "t", "yes", "y"}
    false_values = {"0", "false", "f", "no", "n"}

    normalized = series.astype("string").str.strip().str.lower()
    converted = normalized.map(
        lambda value: 1.0
        if value in true_values
        else 0.0
        if value in false_values
        else np.nan
    )
    return converted.astype("float64")


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Compute a ratio while guarding against zero or invalid denominators.

    Args:
        numerator: Numerator series.
        denominator: Denominator series.

    Returns:
        Ratio series with invalid divisions represented as NaN.
    """
    numerator_values = pd.to_numeric(numerator, errors="coerce").astype("float64")
    denominator_values = pd.to_numeric(denominator, errors="coerce").astype("float64")
    safe_denominator = denominator_values.where(denominator_values > 0)
    ratio = numerator_values / safe_denominator
    return ratio.replace([np.inf, -np.inf], np.nan)


def add_group_lag_features(
    dataframe: pd.DataFrame,
    source_column: str,
    lags: Sequence[int],
    feature_prefix: str,
    group_columns: Sequence[str] = GROUP_COLUMNS,
) -> list[str]:
    """Add grouped lag features for a source column.

    Args:
        dataframe: Input dataframe.
        source_column: Source column used for lagging.
        lags: Lag periods.
        feature_prefix: Feature name prefix.
        group_columns: Grouping columns.

    Returns:
        Generated feature names.
    """
    grouped = dataframe.groupby(list(group_columns), sort=False)[source_column]
    generated: list[str] = []

    for lag in sorted({int(value) for value in lags if int(value) > 0}):
        feature_name = f"{feature_prefix}_lag_{lag}"
        dataframe[feature_name] = grouped.shift(lag)
        generated.append(feature_name)

    return generated


def _apply_shifted_rolling(
    series: pd.Series,
    window: int,
    statistic: str,
    shift_periods: int,
) -> pd.Series:
    """Apply shifted rolling statistics to a single grouped series."""
    shifted = series.shift(shift_periods)
    rolling = shifted.rolling(window=window, min_periods=window)

    if statistic == "mean":
        return rolling.mean()
    if statistic == "std":
        return rolling.std()
    if statistic == "min":
        return rolling.min()
    if statistic == "max":
        return rolling.max()

    raise ValueError(f"Unsupported rolling statistic: {statistic}")


def add_group_shifted_rolling_features(
    dataframe: pd.DataFrame,
    source_column: str,
    window_to_statistics: Mapping[int, Sequence[str]],
    feature_prefix: str,
    group_columns: Sequence[str] = GROUP_COLUMNS,
    shift_periods: int = 1,
) -> list[str]:
    """Add grouped rolling features with an explicit positive shift.

    Args:
        dataframe: Input dataframe.
        source_column: Source column for rolling aggregation.
        window_to_statistics: Mapping of rolling windows to requested statistics.
        feature_prefix: Feature name prefix.
        group_columns: Grouping columns.
        shift_periods: Positive shift applied before rolling.

    Returns:
        Generated feature names.
    """
    if shift_periods <= 0:
        raise ValueError("shift_periods must be positive to preserve leakage safety")

    generated: list[str] = []
    grouped = dataframe.groupby(list(group_columns), sort=False)[source_column]

    for window in sorted(window_to_statistics):
        if window <= 0:
            raise ValueError("Rolling window values must be positive")

        statistics = window_to_statistics[window]
        for statistic in statistics:
            feature_name = f"{feature_prefix}_roll_{statistic}_{window}"
            dataframe[feature_name] = grouped.transform(
                lambda series, current_window=window, current_statistic=statistic: _apply_shifted_rolling(
                    series,
                    current_window,
                    current_statistic,
                    shift_periods,
                )
            )
            generated.append(feature_name)

    return generated


def trim_warmup_rows(
    dataframe: pd.DataFrame,
    required_columns: Sequence[str],
) -> tuple[pd.DataFrame, int]:
    """Drop rows with incomplete warmup history in required feature columns.

    Args:
        dataframe: Input dataframe.
        required_columns: Columns that must be non-null after warmup.

    Returns:
        Tuple of trimmed dataframe and dropped row count.
    """
    validate_feature_input_columns(dataframe, required_columns)

    valid_mask = dataframe[list(required_columns)].notna().all(axis=1)
    dropped = int((~valid_mask).sum())
    trimmed = dataframe.loc[valid_mask].reset_index(drop=True)
    return trimmed, dropped


def ordered_unique(values: Sequence[str]) -> list[str]:
    """Return unique string values preserving input order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered
