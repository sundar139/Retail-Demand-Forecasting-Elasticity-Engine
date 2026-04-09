"""Schema definitions and column normalization helpers."""

from dataclasses import dataclass
import re
from typing import Final

DATE_COLUMN: Final[str] = "date"
STORE_COLUMN: Final[str] = "store_id"
PRODUCT_COLUMN: Final[str] = "product_id"
UNITS_COLUMN: Final[str] = "units_sold"
PRICE_COLUMN: Final[str] = "price"

# Compatibility alias for earlier modules that referenced sku_id.
SKU_COLUMN: Final[str] = PRODUCT_COLUMN

REQUIRED_CANONICAL_COLUMNS: Final[tuple[str, ...]] = (
    DATE_COLUMN,
    STORE_COLUMN,
    PRODUCT_COLUMN,
    UNITS_COLUMN,
    PRICE_COLUMN,
)

OPTIONAL_CANONICAL_COLUMNS: Final[tuple[str, ...]] = (
    "competitor_price",
    "promotion",
    "discount",
    "holiday",
    "weather",
    "inventory_level",
    "demand_forecast",
)

NUMERIC_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (UNITS_COLUMN, PRICE_COLUMN)
NUMERIC_OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "competitor_price",
    "discount",
    "inventory_level",
    "demand_forecast",
)
BOOLEAN_OPTIONAL_COLUMNS: Final[tuple[str, ...]] = ("promotion", "holiday")

GROUP_COLUMNS: Final[tuple[str, str]] = (STORE_COLUMN, PRODUCT_COLUMN)

REQUIRED_SALES_COLUMNS: Final[tuple[str, ...]] = REQUIRED_CANONICAL_COLUMNS

REQUIRED_PRICE_PLAN_COLUMNS: Final[tuple[str, ...]] = (
    DATE_COLUMN,
    STORE_COLUMN,
    PRODUCT_COLUMN,
    PRICE_COLUMN,
)

CANONICAL_ALIAS_MAP: Final[dict[str, str]] = {
    DATE_COLUMN: DATE_COLUMN,
    "sale_date": DATE_COLUMN,
    "sales_date": DATE_COLUMN,
    "transaction_date": DATE_COLUMN,
    "date_sold": DATE_COLUMN,
    STORE_COLUMN: STORE_COLUMN,
    "store": STORE_COLUMN,
    "branch": STORE_COLUMN,
    "branch_id": STORE_COLUMN,
    PRODUCT_COLUMN: PRODUCT_COLUMN,
    "sku": PRODUCT_COLUMN,
    "item_id": PRODUCT_COLUMN,
    "item": PRODUCT_COLUMN,
    "product": PRODUCT_COLUMN,
    "product_code": PRODUCT_COLUMN,
    "sales": UNITS_COLUMN,
    "quantity_sold": UNITS_COLUMN,
    "quantity": UNITS_COLUMN,
    "qty": UNITS_COLUMN,
    "units": UNITS_COLUMN,
    UNITS_COLUMN: UNITS_COLUMN,
    PRICE_COLUMN: PRICE_COLUMN,
    "unit_price": PRICE_COLUMN,
    "selling_price": PRICE_COLUMN,
    "competitorprice": "competitor_price",
    "competitor_price": "competitor_price",
    "promo": "promotion",
    "promotion": "promotion",
    "is_promotion": "promotion",
    "is_holiday": "holiday",
    "holiday": "holiday",
    "discount_pct": "discount",
    "discount": "discount",
    "weather": "weather",
    "inventory": "inventory_level",
    "inventory_level": "inventory_level",
    "forecast": "demand_forecast",
    "demand_forecast": "demand_forecast",
}


def normalize_column_name(column_name: str) -> str:
    """Normalize an arbitrary incoming column name into snake_case.

    Args:
        column_name: Raw incoming column name.

    Returns:
        Normalized snake_case string.
    """
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", column_name.strip())
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_").lower()
    return normalized


def map_to_canonical_column(normalized_name: str) -> str:
    """Map a normalized input name to canonical naming where possible.

    Args:
        normalized_name: Snake_case normalized name.

    Returns:
        Canonical column name if mapped; otherwise the original normalized name.
    """
    return CANONICAL_ALIAS_MAP.get(normalized_name, normalized_name)


@dataclass(frozen=True, slots=True)
class RetailSchema:
    """Canonical retail data schema."""

    required_columns: tuple[str, ...] = REQUIRED_CANONICAL_COLUMNS
    optional_columns: tuple[str, ...] = OPTIONAL_CANONICAL_COLUMNS
