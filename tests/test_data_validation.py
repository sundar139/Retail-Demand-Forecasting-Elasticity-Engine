"""Tests for data schema validation and type coercion."""

import pandas as pd
import pytest

from retail_forecasting.data_validation import DataValidationError, validate_and_standardize_dataframe


def test_missing_required_column_raises_actionable_error() -> None:
    """Validation should fail clearly when required columns are missing."""
    frame = pd.DataFrame(
        {
            "sale_date": ["2024-01-01"],
            "store": ["S1"],
            "sku": ["P1"],
            "sales": [10],
        }
    )

    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_and_standardize_dataframe(frame)


def test_alias_mapping_and_type_coercion() -> None:
    """Aliases should map correctly and required fields should be coerced safely."""
    frame = pd.DataFrame(
        {
            "sale_date": ["2024-01-01", "2024-01-02"],
            "branch": ["S1", "S1"],
            "item_id": ["P1", "P1"],
            "quantity_sold": ["10", "0"],
            "unit_price": ["9.5", "9.0"],
            "promotion": ["yes", "no"],
            "discount": ["0.10", "0.00"],
            "weather": ["sunny", "rain"],
        }
    )

    validated, optional_columns = validate_and_standardize_dataframe(frame)

    assert set(["date", "store_id", "product_id", "units_sold", "price"]).issubset(validated.columns)
    assert pd.api.types.is_datetime64_any_dtype(validated["date"])
    assert pd.api.types.is_numeric_dtype(validated["units_sold"])
    assert pd.api.types.is_numeric_dtype(validated["price"])
    assert "promotion" in optional_columns
    assert "discount" in optional_columns
    assert "weather" in optional_columns


def test_optional_columns_preserved_when_present() -> None:
    """Optional columns should remain in the standardized frame."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "store_id": ["S1"],
            "product_id": ["P1"],
            "units_sold": [10],
            "price": [5.0],
            "inventory_level": [100],
            "demand_forecast": [12.0],
        }
    )

    validated, optional_columns = validate_and_standardize_dataframe(frame)

    assert "inventory_level" in validated.columns
    assert "demand_forecast" in validated.columns
    assert "inventory_level" in optional_columns
    assert "demand_forecast" in optional_columns
