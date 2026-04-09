"""Schema validation and deterministic type coercion for retail datasets."""

from collections import defaultdict
from collections.abc import Iterable
import logging

import pandas as pd

from retail_forecasting.schemas import (
    BOOLEAN_OPTIONAL_COLUMNS,
    DATE_COLUMN,
    NUMERIC_OPTIONAL_COLUMNS,
    NUMERIC_REQUIRED_COLUMNS,
    OPTIONAL_CANONICAL_COLUMNS,
    PRODUCT_COLUMN,
    REQUIRED_CANONICAL_COLUMNS,
    STORE_COLUMN,
    map_to_canonical_column,
    normalize_column_name,
)

LOGGER = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when an input dataset fails schema or value validation."""


def _check_column_collisions(mapped_columns: list[str], original_columns: list[str]) -> None:
    """Detect collisions where multiple raw columns map to one canonical name.

    Args:
        mapped_columns: Canonical/normalized column names.
        original_columns: Original column names before normalization.

    Raises:
        DataValidationError: If collisions are found.
    """
    reverse_lookup: dict[str, list[str]] = defaultdict(list)
    for source, target in zip(original_columns, mapped_columns, strict=True):
        reverse_lookup[target].append(source)

    collisions = {target: sources for target, sources in reverse_lookup.items() if len(sources) > 1}
    if collisions:
        collision_text = "; ".join(
            f"{target} <- {', '.join(sources)}" for target, sources in collisions.items()
        )
        raise DataValidationError(
            "Multiple source columns map to the same canonical name. "
            f"Please keep only one per field: {collision_text}"
        )


def normalize_and_map_columns(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Normalize incoming columns to snake_case and map aliases to canonical names.

    Args:
        dataframe: Raw input dataframe.

    Returns:
        Tuple of renamed dataframe and source-to-target mapping.
    """
    source_columns = [str(column) for column in dataframe.columns]
    normalized = [normalize_column_name(column) for column in source_columns]
    mapped = [map_to_canonical_column(column) for column in normalized]

    _check_column_collisions(mapped, source_columns)
    mapping = {source: target for source, target in zip(source_columns, mapped, strict=True)}
    renamed = dataframe.rename(columns=mapping)
    return renamed, mapping


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    """Ensure all required canonical columns are present.

    Args:
        dataframe: Canonicalized dataframe.

    Raises:
        DataValidationError: If required columns are missing.
    """
    missing = sorted(set(REQUIRED_CANONICAL_COLUMNS) - set(dataframe.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise DataValidationError(
            "Dataset is missing required columns: "
            f"{missing_text}. Please ensure the Kaggle CSV includes these fields or aliases."
        )


def _coerce_datetime_column(dataframe: pd.DataFrame) -> None:
    """Convert the canonical date column to datetime.

    Args:
        dataframe: Input dataframe in canonical naming.

    Raises:
        DataValidationError: If date parsing fails.
    """
    parsed = pd.to_datetime(dataframe[DATE_COLUMN], errors="coerce")
    if parsed.isna().any():
        invalid_count = int(parsed.isna().sum())
        raise DataValidationError(
            f"Column '{DATE_COLUMN}' contains {invalid_count} invalid date values. "
            "Fix date formatting in the source CSV (e.g. YYYY-MM-DD)."
        )
    dataframe[DATE_COLUMN] = parsed.dt.floor("D")


def _coerce_numeric_columns(dataframe: pd.DataFrame, columns: Iterable[str], required: bool) -> None:
    """Coerce selected columns to numeric values.

    Args:
        dataframe: Input dataframe.
        columns: Columns to coerce.
        required: Whether NaN values should trigger validation failure.

    Raises:
        DataValidationError: If required numeric conversion fails.
    """
    for column in columns:
        if column not in dataframe.columns:
            continue
        converted = pd.to_numeric(dataframe[column], errors="coerce")
        if required and converted.isna().any():
            invalid_count = int(converted.isna().sum())
            raise DataValidationError(
                f"Column '{column}' has {invalid_count} values that cannot be parsed as numeric."
            )
        dataframe[column] = converted


def _coerce_boolean_like_columns(dataframe: pd.DataFrame) -> None:
    """Normalize optional boolean-like columns to pandas boolean dtype.

    Args:
        dataframe: Input dataframe.
    """
    true_values = {"1", "true", "t", "yes", "y"}
    false_values = {"0", "false", "f", "no", "n"}

    for column in BOOLEAN_OPTIONAL_COLUMNS:
        if column not in dataframe.columns:
            continue
        normalized = dataframe[column].astype("string").str.strip().str.lower()
        converted = normalized.map(
            lambda value: True
            if value in true_values
            else False
            if value in false_values
            else pd.NA
        )
        dataframe[column] = pd.Series(converted, dtype="boolean")


def _validate_identifier_columns(dataframe: pd.DataFrame) -> None:
    """Validate store_id and product_id fields.

    Args:
        dataframe: Input dataframe.

    Raises:
        DataValidationError: If identifier values are missing.
    """
    for column in (STORE_COLUMN, PRODUCT_COLUMN):
        normalized = dataframe[column].astype("string").str.strip()
        invalid_mask = normalized.isna() | (normalized == "")
        if invalid_mask.any():
            invalid_count = int(invalid_mask.sum())
            raise DataValidationError(
                f"Column '{column}' has {invalid_count} missing or empty values."
            )
        dataframe[column] = normalized


def _validate_core_value_ranges(dataframe: pd.DataFrame) -> None:
    """Validate numeric business constraints for required fields.

    Args:
        dataframe: Input dataframe.

    Raises:
        DataValidationError: If numeric constraints are violated.
    """
    if (dataframe["units_sold"] < 0).any():
        count = int((dataframe["units_sold"] < 0).sum())
        raise DataValidationError(f"Column 'units_sold' has {count} negative values.")

    if (dataframe["price"] <= 0).any():
        count = int((dataframe["price"] <= 0).sum())
        raise DataValidationError(f"Column 'price' has {count} non-positive values.")


def validate_and_standardize_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Normalize column names, validate schema, and coerce deterministic types.

    Args:
        dataframe: Raw dataframe loaded from CSV.

    Returns:
        Tuple of validated dataframe and optional columns present.

    Raises:
        DataValidationError: If validation fails.
    """
    standardized, _ = normalize_and_map_columns(dataframe.copy(deep=True))
    validate_required_columns(standardized)

    _coerce_datetime_column(standardized)
    _coerce_numeric_columns(standardized, NUMERIC_REQUIRED_COLUMNS, required=True)
    _coerce_numeric_columns(standardized, NUMERIC_OPTIONAL_COLUMNS, required=False)
    _coerce_boolean_like_columns(standardized)
    _validate_identifier_columns(standardized)
    _validate_core_value_ranges(standardized)

    optional_columns = sorted(set(OPTIONAL_CANONICAL_COLUMNS) & set(standardized.columns))
    LOGGER.info(
        "Validated schema with %d rows, %d columns, %d optional columns",
        standardized.shape[0],
        standardized.shape[1],
        len(optional_columns),
    )
    return standardized, optional_columns
