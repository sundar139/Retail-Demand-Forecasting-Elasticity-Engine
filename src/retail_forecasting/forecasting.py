"""Demand forecasting logic using fitted price elasticities."""

import logging

import pandas as pd

from retail_forecasting.preprocessing import validate_sales_dataframe
from retail_forecasting.schemas import (
    DATE_COLUMN,
    GROUP_COLUMNS,
    PRICE_COLUMN,
    REQUIRED_PRICE_PLAN_COLUMNS,
    SKU_COLUMN,
    STORE_COLUMN,
    UNITS_COLUMN,
)

LOGGER = logging.getLogger(__name__)


def _validate_price_plan_columns(price_plan: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a future price plan dataframe.

    Args:
        price_plan: Raw price plan dataframe.

    Returns:
        Cleaned price plan dataframe.

    Raises:
        ValueError: If the dataframe is malformed.
    """
    missing = sorted(set(REQUIRED_PRICE_PLAN_COLUMNS) - set(price_plan.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required price plan columns: {missing_text}")

    plan = price_plan.copy(deep=True)
    plan[DATE_COLUMN] = pd.to_datetime(plan[DATE_COLUMN], errors="coerce").dt.floor("D")
    plan[PRICE_COLUMN] = pd.to_numeric(plan[PRICE_COLUMN], errors="coerce")

    if plan[DATE_COLUMN].isna().any():
        raise ValueError("Price plan contains invalid date values")
    if plan[PRICE_COLUMN].isna().any() or (plan[PRICE_COLUMN] <= 0.0).any():
        raise ValueError("Price plan contains invalid price values")

    plan[STORE_COLUMN] = plan[STORE_COLUMN].astype("string")
    plan[SKU_COLUMN] = plan[SKU_COLUMN].astype("string")
    return plan


def _build_price_lookup(price_plan: pd.DataFrame | None) -> dict[tuple[str, str], dict[pd.Timestamp, float]]:
    """Build a deterministic lookup structure for planned prices.

    Args:
        price_plan: Optional cleaned price-plan dataframe.

    Returns:
        Mapping keyed by (store_id, sku_id) then future date.
    """
    if price_plan is None or price_plan.empty:
        return {}

    lookup: dict[tuple[str, str], dict[pd.Timestamp, float]] = {}
    for (store_id, sku_id), group_df in price_plan.groupby(list(GROUP_COLUMNS), sort=True):
        nested = {
            pd.Timestamp(date_value): float(price_value)
            for date_value, price_value in zip(
                group_df[DATE_COLUMN].tolist(),
                group_df[PRICE_COLUMN].tolist(),
                strict=True,
            )
        }
        lookup[(str(store_id), str(sku_id))] = nested

    return lookup


def _build_elasticity_map(elasticity_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Build elasticity lookup keyed by store and SKU.

    Args:
        elasticity_df: Dataframe containing fitted elasticity values.

    Returns:
        Dictionary mapping (store_id, sku_id) to elasticity.
    """
    if elasticity_df.empty:
        return {}

    required_columns = {STORE_COLUMN, SKU_COLUMN, "price_elasticity"}
    missing = sorted(required_columns - set(elasticity_df.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Elasticity frame is missing columns: {missing_text}")

    elasticity_map: dict[tuple[str, str], float] = {}
    for row in elasticity_df.itertuples(index=False):
        key = (str(getattr(row, STORE_COLUMN)), str(getattr(row, SKU_COLUMN)))
        elasticity_map[key] = float(getattr(row, "price_elasticity"))

    return elasticity_map


def forecast_with_price_plan(
    history_df: pd.DataFrame,
    elasticity_df: pd.DataFrame,
    horizon_days: int,
    lookback_days: int = 28,
    price_plan: pd.DataFrame | None = None,
    default_elasticity: float = -1.0,
) -> pd.DataFrame:
    """Create demand forecasts from recent history and elasticity values.

    Args:
        history_df: Historical cleaned sales data.
        elasticity_df: Fitted elasticity model results.
        horizon_days: Number of daily periods to forecast.
        lookback_days: Number of trailing historical days for baseline demand.
        price_plan: Optional future prices by date/store/SKU.
        default_elasticity: Elasticity used when no model exists for a group.

    Returns:
        Forecast dataframe with one row per future date and group.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")

    history = validate_sales_dataframe(history_df)
    cleaned_plan = _validate_price_plan_columns(price_plan) if price_plan is not None else None
    price_lookup = _build_price_lookup(cleaned_plan)
    elasticity_map = _build_elasticity_map(elasticity_df)

    rows: list[dict[str, object]] = []

    for (store_id, sku_id), group_df in history.groupby(list(GROUP_COLUMNS), sort=True):
        ordered_group = group_df.sort_values(DATE_COLUMN, kind="mergesort")
        trailing = ordered_group.tail(lookback_days)

        baseline_units = float(trailing[UNITS_COLUMN].mean())
        reference_price = float(trailing[PRICE_COLUMN].mean())
        last_date = pd.Timestamp(trailing[DATE_COLUMN].max())

        key = (str(store_id), str(sku_id))
        elasticity = float(elasticity_map.get(key, default_elasticity))
        planned_prices = price_lookup.get(key, {})

        for day_offset in range(1, horizon_days + 1):
            forecast_date = last_date + pd.Timedelta(days=day_offset)
            planned_price = float(planned_prices.get(forecast_date, reference_price))
            price_ratio = planned_price / reference_price
            forecast_units = max(baseline_units * (price_ratio ** elasticity), 0.0)

            rows.append(
                {
                    STORE_COLUMN: str(store_id),
                    SKU_COLUMN: str(sku_id),
                    DATE_COLUMN: forecast_date,
                    "forecast_units": forecast_units,
                    "baseline_units": baseline_units,
                    "planned_price": planned_price,
                    "reference_price": reference_price,
                    "applied_elasticity": elasticity,
                }
            )

    forecast_df = pd.DataFrame.from_records(rows)
    ordered_forecast = forecast_df.sort_values(
        by=[STORE_COLUMN, SKU_COLUMN, DATE_COLUMN], kind="mergesort"
    ).reset_index(drop=True)

    LOGGER.info("Generated %d forecast rows", len(ordered_forecast))
    return ordered_forecast
