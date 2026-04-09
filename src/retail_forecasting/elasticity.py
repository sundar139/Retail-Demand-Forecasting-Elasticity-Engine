"""Price elasticity estimation for retail demand."""

from dataclasses import asdict, dataclass
import logging

import numpy as np
import pandas as pd

from retail_forecasting.preprocessing import validate_sales_dataframe
from retail_forecasting.schemas import GROUP_COLUMNS, PRICE_COLUMN, SKU_COLUMN, STORE_COLUMN, UNITS_COLUMN

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ElasticityEstimate:
    """Represents a fitted log-log demand elasticity model for one SKU and store."""

    store_id: str
    sku_id: str
    intercept: float
    price_elasticity: float
    r_squared: float
    n_observations: int


def _fit_log_log_regression(store_id: str, sku_id: str, group_df: pd.DataFrame) -> ElasticityEstimate:
    """Fit log(units_sold) = intercept + beta * log(price) for one group.

    Args:
        store_id: Store identifier.
        sku_id: SKU identifier.
        group_df: Group-specific rows with strictly positive units and prices.

    Returns:
        ElasticityEstimate for the group.
    """
    log_price = np.log(group_df[PRICE_COLUMN].to_numpy(dtype=float))
    log_units = np.log(group_df[UNITS_COLUMN].to_numpy(dtype=float))

    design_matrix = np.column_stack((np.ones_like(log_price), log_price))
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, log_units, rcond=None)

    intercept = float(coefficients[0])
    price_elasticity = float(coefficients[1])

    fitted = design_matrix @ coefficients
    residual_sum_squares = float(np.sum((log_units - fitted) ** 2))
    total_sum_squares = float(np.sum((log_units - np.mean(log_units)) ** 2))
    r_squared = 0.0 if total_sum_squares == 0.0 else 1.0 - (residual_sum_squares / total_sum_squares)

    return ElasticityEstimate(
        store_id=store_id,
        sku_id=sku_id,
        intercept=intercept,
        price_elasticity=price_elasticity,
        r_squared=r_squared,
        n_observations=int(len(group_df)),
    )


def fit_elasticity_models(sales_df: pd.DataFrame, min_observations: int = 8) -> pd.DataFrame:
    """Fit log-log price elasticity models for each store and SKU pair.

    Args:
        sales_df: Input sales dataframe.
        min_observations: Minimum required rows for fitting each group model.

    Returns:
        Dataframe containing one fitted model per eligible group.
    """
    if min_observations < 2:
        raise ValueError("min_observations must be at least 2")

    cleaned = validate_sales_dataframe(sales_df)
    group_columns = list(GROUP_COLUMNS)
    estimates: list[ElasticityEstimate] = []

    for (store_id, sku_id), group_df in cleaned.groupby(group_columns, sort=True):
        usable = group_df[group_df[UNITS_COLUMN] > 0.0]
        if len(usable) < min_observations:
            LOGGER.warning(
                "Skipping elasticity fit for store=%s sku=%s due to insufficient rows (%d < %d)",
                store_id,
                sku_id,
                len(usable),
                min_observations,
            )
            continue

        estimate = _fit_log_log_regression(
            store_id=str(store_id),
            sku_id=str(sku_id),
            group_df=usable,
        )
        estimates.append(estimate)

    if not estimates:
        return pd.DataFrame(
            columns=[
                STORE_COLUMN,
                SKU_COLUMN,
                "intercept",
                "price_elasticity",
                "r_squared",
                "n_observations",
            ]
        )

    result = pd.DataFrame([asdict(record) for record in estimates])
    ordered = result.sort_values(by=[STORE_COLUMN, SKU_COLUMN], kind="mergesort").reset_index(drop=True)

    LOGGER.info("Fitted %d elasticity models", len(ordered))
    return ordered
