"""Tests for schema normalization and alias mapping."""

from retail_forecasting.schemas import map_to_canonical_column, normalize_column_name


def test_alias_mapping_to_canonical_columns() -> None:
    """Likely alias names should map to required canonical names."""
    assert map_to_canonical_column(normalize_column_name("Sale Date")) == "date"
    assert map_to_canonical_column(normalize_column_name("store")) == "store_id"
    assert map_to_canonical_column(normalize_column_name("sku")) == "product_id"
    assert map_to_canonical_column(normalize_column_name("Quantity Sold")) == "units_sold"
    assert map_to_canonical_column(normalize_column_name("unit_price")) == "price"


def test_column_normalization_snake_case() -> None:
    """Incoming column names should normalize to deterministic snake_case."""
    assert normalize_column_name("Competitor Price") == "competitor_price"
    assert normalize_column_name("Store-ID") == "store_id"
    assert normalize_column_name("DemandForecast") == "demand_forecast"
