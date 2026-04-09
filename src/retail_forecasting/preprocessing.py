"""Phase 2 preprocessing pipeline for ingestion, cleaning, and chronological splitting."""

from collections.abc import Sequence
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from retail_forecasting.data_loading import discover_and_load_csv
from retail_forecasting.data_validation import validate_and_standardize_dataframe
from retail_forecasting.paths import build_project_paths
from retail_forecasting.schemas import (
    DATE_COLUMN,
    GROUP_COLUMNS,
    OPTIONAL_CANONICAL_COLUMNS,
    PRICE_COLUMN,
    PRODUCT_COLUMN,
    STORE_COLUMN,
    UNITS_COLUMN,
)

LOGGER = logging.getLogger(__name__)


def _to_iso_date(value: pd.Timestamp | None) -> str | None:
    """Convert timestamps to ISO date strings.

    Args:
        value: Timestamp value.

    Returns:
        ISO date string or None.
    """
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def remove_exact_duplicates(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate rows in a deterministic way.

    Args:
        dataframe: Input dataframe.

    Returns:
        Tuple of deduplicated dataframe and removed row count.
    """
    duplicate_mask = dataframe.duplicated(keep="first")
    removed_count = int(duplicate_mask.sum())
    deduplicated = dataframe.loc[~duplicate_mask].reset_index(drop=True)
    return deduplicated, removed_count


def sort_retail_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by store, product, and date for deterministic ordering.

    Args:
        dataframe: Input dataframe.

    Returns:
        Sorted dataframe.
    """
    return dataframe.sort_values(
        by=[STORE_COLUMN, PRODUCT_COLUMN, DATE_COLUMN],
        kind="mergesort",
    ).reset_index(drop=True)


def write_parquet(dataframe: pd.DataFrame, output_path: Path) -> Path:
    """Write a dataframe to parquet.

    Args:
        dataframe: Dataframe to write.
        output_path: Destination parquet path.

    Returns:
        Destination path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)
    return output_path


def write_json(payload: dict[str, object], output_path: Path) -> Path:
    """Write a JSON payload to disk with deterministic formatting.

    Args:
        payload: JSON-serializable payload.
        output_path: Destination file path.

    Returns:
        Destination path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def build_data_summary(
    cleaned_df: pd.DataFrame,
    source_filename: str,
    duplicate_rows_removed: int,
    optional_columns_detected: list[str],
) -> dict[str, object]:
    """Build a machine-readable summary artifact for cleaned data.

    Args:
        cleaned_df: Cleaned canonical dataframe.
        source_filename: Source CSV filename.
        duplicate_rows_removed: Number of exact duplicate rows removed.
        optional_columns_detected: Optional canonical columns present.

    Returns:
        Summary payload.
    """
    min_date = _to_iso_date(pd.Timestamp(cleaned_df[DATE_COLUMN].min())) if not cleaned_df.empty else None
    max_date = _to_iso_date(pd.Timestamp(cleaned_df[DATE_COLUMN].max())) if not cleaned_df.empty else None

    summary = {
        "source_filename": source_filename,
        "total_row_count": int(cleaned_df.shape[0]),
        "total_column_count": int(cleaned_df.shape[1]),
        "min_date": min_date,
        "max_date": max_date,
        "number_of_stores": int(cleaned_df[STORE_COLUMN].nunique(dropna=True)),
        "number_of_products": int(cleaned_df[PRODUCT_COLUMN].nunique(dropna=True)),
        "missing_value_counts_by_column": {
            column: int(count) for column, count in cleaned_df.isna().sum().to_dict().items()
        },
        "duplicate_row_count_removed": int(duplicate_rows_removed),
        "zero_demand_rate": float((cleaned_df[UNITS_COLUMN] == 0).mean()) if not cleaned_df.empty else 0.0,
        "optional_columns_detected": sorted(optional_columns_detected),
    }
    return summary


def clean_retail_dataframe(
    raw_df: pd.DataFrame,
    source_filename: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate, clean, and summarize a raw retail dataset.

    Args:
        raw_df: Raw dataframe loaded from CSV.
        source_filename: Source CSV filename.

    Returns:
        Tuple of cleaned dataframe and summary payload.
    """
    standardized_df, optional_columns = validate_and_standardize_dataframe(raw_df)
    deduplicated_df, duplicate_rows_removed = remove_exact_duplicates(standardized_df)
    cleaned_df = sort_retail_rows(deduplicated_df)

    summary = build_data_summary(
        cleaned_df=cleaned_df,
        source_filename=source_filename,
        duplicate_rows_removed=duplicate_rows_removed,
        optional_columns_detected=optional_columns,
    )
    LOGGER.info(
        "Cleaned dataset rows=%d duplicates_removed=%d",
        cleaned_df.shape[0],
        duplicate_rows_removed,
    )
    return cleaned_df, summary


def _validate_split_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    """Validate split ratios.

    Args:
        train_ratio: Train ratio.
        validation_ratio: Validation ratio.
        test_ratio: Test ratio.

    Raises:
        ValueError: If ratios are invalid.
    """
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("Split ratios must all be positive.")

    ratio_sum = sum(ratios)
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, received {ratio_sum:.6f}."
        )


def _derive_split_cutoffs(
    unique_dates: list[pd.Timestamp],
    train_ratio: float,
    validation_ratio: float,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Derive validation and test start dates from ratio-based boundaries.

    Args:
        unique_dates: Sorted unique dates.
        train_ratio: Train ratio.
        validation_ratio: Validation ratio.

    Returns:
        Tuple of validation start and test start dates.
    """
    total_dates = len(unique_dates)
    if total_dates < 3:
        raise ValueError("At least 3 distinct dates are required for train/validation/test splits.")

    train_cut_index = max(1, int(total_dates * train_ratio))
    validation_cut_index = max(train_cut_index + 1, int(total_dates * (train_ratio + validation_ratio)))
    validation_cut_index = min(validation_cut_index, total_dates - 1)
    train_cut_index = min(train_cut_index, validation_cut_index - 1)

    validation_start = unique_dates[train_cut_index]
    test_start = unique_dates[validation_cut_index]
    return validation_start, test_start


def split_chronologically(
    cleaned_df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    validation_start: str | None = None,
    test_start: str | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Create deterministic chronological train/validation/test splits.

    Args:
        cleaned_df: Cleaned canonical dataframe.
        train_ratio: Train ratio for ratio-based splitting.
        validation_ratio: Validation ratio for ratio-based splitting.
        test_ratio: Test ratio for ratio-based splitting.
        validation_start: Optional explicit validation start date.
        test_start: Optional explicit test start date.

    Returns:
        Tuple of split dataframes and split summary payload.
    """
    if cleaned_df.empty:
        raise ValueError("Cannot split an empty dataframe.")

    ordered = sort_retail_rows(cleaned_df)
    unique_dates = sorted(pd.to_datetime(ordered[DATE_COLUMN]).dt.floor("D").unique().tolist())

    if validation_start is not None or test_start is not None:
        if validation_start is None or test_start is None:
            raise ValueError("Both validation_start and test_start must be provided together.")
        validation_start_ts = pd.to_datetime(validation_start, errors="raise").floor("D")
        test_start_ts = pd.to_datetime(test_start, errors="raise").floor("D")
    else:
        _validate_split_ratios(train_ratio, validation_ratio, test_ratio)
        validation_start_ts, test_start_ts = _derive_split_cutoffs(
            unique_dates=unique_dates,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )

    if validation_start_ts >= test_start_ts:
        raise ValueError("validation_start must be earlier than test_start.")

    train_df = ordered[ordered[DATE_COLUMN] < validation_start_ts].reset_index(drop=True)
    validation_df = ordered[
        (ordered[DATE_COLUMN] >= validation_start_ts) & (ordered[DATE_COLUMN] < test_start_ts)
    ].reset_index(drop=True)
    test_df = ordered[ordered[DATE_COLUMN] >= test_start_ts].reset_index(drop=True)

    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError(
            "Chronological split produced an empty partition. "
            "Adjust split ratios or explicit cutoff dates."
        )

    total_rows = int(len(ordered))

    def split_metrics(frame: pd.DataFrame) -> dict[str, object]:
        return {
            "row_count": int(len(frame)),
            "share_of_total_rows": float(len(frame) / total_rows),
            "min_date": _to_iso_date(pd.Timestamp(frame[DATE_COLUMN].min())),
            "max_date": _to_iso_date(pd.Timestamp(frame[DATE_COLUMN].max())),
        }

    summary = {
        "total_rows": total_rows,
        "splits": {
            "train": split_metrics(train_df),
            "validation": split_metrics(validation_df),
            "test": split_metrics(test_df),
        },
        "cutoffs": {
            "validation_start": _to_iso_date(validation_start_ts),
            "test_start": _to_iso_date(test_start_ts),
        },
    }

    return {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }, summary


def prepare_data_pipeline(
    input_path: Path | None = None,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    validation_start: str | None = None,
    test_start: str | None = None,
) -> dict[str, Path]:
    """Run the full ingestion, cleaning, summary, and split workflow.

    Args:
        input_path: Optional input CSV path.
        train_ratio: Ratio-based train split.
        validation_ratio: Ratio-based validation split.
        test_ratio: Ratio-based test split.
        validation_start: Optional explicit validation start date.
        test_start: Optional explicit test start date.

    Returns:
        Paths to generated artifacts.
    """
    paths = build_project_paths()
    source_path, raw_df = discover_and_load_csv(input_path=input_path, raw_dir=paths.data_raw_dir)

    cleaned_df, summary = clean_retail_dataframe(raw_df=raw_df, source_filename=source_path.name)
    cleaned_path = write_parquet(cleaned_df, paths.data_interim_dir / "cleaned_retail.parquet")
    summary_path = write_json(summary, paths.artifacts_dir / "data_summary.json")

    split_frames, split_summary = split_chronologically(
        cleaned_df,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        validation_start=validation_start,
        test_start=test_start,
    )
    train_path = write_parquet(split_frames["train"], paths.data_processed_dir / "train.parquet")
    validation_path = write_parquet(
        split_frames["validation"], paths.data_processed_dir / "validation.parquet"
    )
    test_path = write_parquet(split_frames["test"], paths.data_processed_dir / "test.parquet")
    split_summary_path = write_json(split_summary, paths.artifacts_dir / "split_summary.json")

    return {
        "source_csv": source_path,
        "cleaned_parquet": cleaned_path,
        "data_summary_json": summary_path,
        "train_parquet": train_path,
        "validation_parquet": validation_path,
        "test_parquet": test_path,
        "split_summary_json": split_summary_path,
    }


def load_json(path: Path) -> dict[str, object]:
    """Load a JSON artifact from disk.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON payload.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def validate_sales_dataframe(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper for validating and ordering retail dataframes.

    Args:
        sales_df: Input dataframe.

    Returns:
        Validated, deduplicated, and sorted dataframe.
    """
    standardized_df, _ = validate_and_standardize_dataframe(sales_df)
    deduplicated, _ = remove_exact_duplicates(standardized_df)
    return sort_retail_rows(deduplicated)


def create_features(
    sales_df: pd.DataFrame,
    lags: Sequence[int] = (1, 7, 14),
    rolling_window: int = 7,
) -> pd.DataFrame:
    """Compatibility feature generator retained for prior forecasting modules.

    Args:
        sales_df: Input dataframe.
        lags: Lag periods for units_sold.
        rolling_window: Rolling window size.

    Returns:
        Feature-enriched dataframe.
    """
    if rolling_window <= 0:
        raise ValueError("rolling_window must be positive")

    cleaned = validate_sales_dataframe(sales_df)
    features = cleaned.copy(deep=True)

    distinct_lags = sorted({int(lag) for lag in lags if int(lag) > 0})
    grouped_units = features.groupby(list(GROUP_COLUMNS), sort=False)[UNITS_COLUMN]
    for lag in distinct_lags:
        features[f"units_lag_{lag}"] = grouped_units.shift(lag)

    features[f"units_roll_mean_{rolling_window}"] = grouped_units.transform(
        lambda series: series.shift(1).rolling(window=rolling_window, min_periods=1).mean()
    )
    features["log_units"] = np.log(features[UNITS_COLUMN] + 1.0)
    features["log_price"] = np.log(features[PRICE_COLUMN])
    features["day_of_week"] = features[DATE_COLUMN].dt.dayofweek.astype("int8")
    features["month"] = features[DATE_COLUMN].dt.month.astype("int8")

    present_optional = sorted(set(OPTIONAL_CANONICAL_COLUMNS) & set(features.columns))
    LOGGER.debug("Feature generation retained optional columns: %s", ", ".join(present_optional))
    return features
