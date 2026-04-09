"""Phase 6 reusable forecasting baselines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from retail_forecasting.features_common import ensure_group_date_sort_order
from retail_forecasting.schemas import DATE_COLUMN, PRODUCT_COLUMN, STORE_COLUMN, UNITS_COLUMN

BASELINE_NAIVE_LAST = "naive_last"
BASELINE_SEASONAL_NAIVE_7 = "seasonal_naive_7"
BASELINE_ROLLING_MEAN_7 = "rolling_mean_7"
BASELINE_ROLLING_MEAN_28 = "rolling_mean_28"
BASELINE_MEDIAN_EXPANDING = "median_expanding"

DEFAULT_BASELINE_MODELS: tuple[str, ...] = (
    BASELINE_NAIVE_LAST,
    BASELINE_SEASONAL_NAIVE_7,
    BASELINE_ROLLING_MEAN_7,
    BASELINE_ROLLING_MEAN_28,
    BASELINE_MEDIAN_EXPANDING,
)


@dataclass(frozen=True, slots=True)
class BaselinePredictionBundle:
    """Container for baseline predictions across validation and test splits."""

    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame

def generate_baseline_predictions(
    history_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    baseline_name: str,
    target_column: str = UNITS_COLUMN,
    date_column: str = DATE_COLUMN,
    segment_columns: Sequence[str] = (STORE_COLUMN, PRODUCT_COLUMN),
) -> pd.DataFrame:
    """Generate one-step-ahead baseline predictions aligned to target rows."""
    if history_frame.empty:
        raise ValueError("history_frame must not be empty for baseline predictions")
    if target_frame.empty:
        return pd.DataFrame(
            columns=[
                date_column,
                STORE_COLUMN,
                PRODUCT_COLUMN,
                "actual",
                "prediction",
                "model_name",
            ]
        )

    required_columns = {date_column, *segment_columns, target_column}
    missing_history = sorted(required_columns - set(history_frame.columns))
    missing_target = sorted(required_columns - set(target_frame.columns))

    if missing_history:
        raise ValueError(
            f"History dataframe is missing required columns: {', '.join(missing_history)}"
        )
    if missing_target:
        raise ValueError(
            f"Target dataframe is missing required columns: {', '.join(missing_target)}"
        )

    history = ensure_group_date_sort_order(history_frame, group_columns=segment_columns, date_column=date_column)
    target = ensure_group_date_sort_order(target_frame, group_columns=segment_columns, date_column=date_column)

    history = history.copy(deep=True)
    target = target.copy(deep=True)
    history["_is_target"] = False
    target["_is_target"] = True
    target["_target_row_order"] = np.arange(len(target), dtype="int64")

    combined = pd.concat([history, target], axis=0, ignore_index=True)
    combined["_target_row_order"] = pd.to_numeric(
        combined.get("_target_row_order", np.nan),
        errors="coerce",
    )

    sort_columns = list(segment_columns) + [date_column, "_is_target", "_target_row_order"]
    combined = combined.sort_values(by=sort_columns, kind="mergesort").reset_index(drop=True)

    numeric_target = pd.to_numeric(combined[target_column], errors="coerce").astype("float64")
    grouped = numeric_target.groupby([combined[column_name] for column_name in segment_columns], sort=False)

    if baseline_name == BASELINE_NAIVE_LAST:
        prediction_series = grouped.shift(1)
    elif baseline_name == BASELINE_SEASONAL_NAIVE_7:
        seasonal = grouped.shift(7)
        fallback = grouped.shift(1)
        prediction_series = seasonal.fillna(fallback)
    elif baseline_name == BASELINE_ROLLING_MEAN_7:
        prediction_series = grouped.transform(
            lambda series: series.shift(1).rolling(window=7, min_periods=1).mean()
        )
    elif baseline_name == BASELINE_ROLLING_MEAN_28:
        prediction_series = grouped.transform(
            lambda series: series.shift(1).rolling(window=28, min_periods=1).mean()
        )
    elif baseline_name == BASELINE_MEDIAN_EXPANDING:
        prediction_series = grouped.transform(
            lambda series: series.shift(1).expanding(min_periods=1).median()
        )
    else:
        allowed = ", ".join(DEFAULT_BASELINE_MODELS)
        raise ValueError(f"Unsupported baseline '{baseline_name}'. Allowed values: {allowed}")

    combined["prediction"] = prediction_series.astype("float64")
    combined["actual"] = numeric_target

    history_median = float(pd.to_numeric(history[target_column], errors="coerce").median(skipna=True))
    if np.isnan(history_median):
        history_median = 0.0

    target_predictions = combined.loc[combined["_is_target"]].copy()
    target_predictions["prediction"] = target_predictions["prediction"].fillna(history_median)

    ordered = target_predictions.sort_values(by=["_target_row_order"], kind="mergesort").reset_index(drop=True)
    output_columns = [date_column, STORE_COLUMN, PRODUCT_COLUMN, "actual", "prediction"]
    output = ordered[output_columns].copy(deep=True)
    output["model_name"] = baseline_name
    return output


def run_baseline_suite(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target_column: str = UNITS_COLUMN,
    baseline_names: Sequence[str] = DEFAULT_BASELINE_MODELS,
) -> BaselinePredictionBundle:
    """Run all configured baselines for validation and test splits."""
    train = ensure_group_date_sort_order(train_frame)
    validation = ensure_group_date_sort_order(validation_frame)
    test = ensure_group_date_sort_order(test_frame)

    history_for_test = pd.concat([train, validation], axis=0, ignore_index=True)

    validation_predictions: list[pd.DataFrame] = []
    test_predictions: list[pd.DataFrame] = []

    for baseline_name in baseline_names:
        validation_pred = generate_baseline_predictions(
            history_frame=train,
            target_frame=validation,
            baseline_name=baseline_name,
            target_column=target_column,
        )
        validation_pred["split"] = "validation"
        validation_pred["model_family"] = "baseline"
        validation_predictions.append(validation_pred)

        test_pred = generate_baseline_predictions(
            history_frame=history_for_test,
            target_frame=test,
            baseline_name=baseline_name,
            target_column=target_column,
        )
        test_pred["split"] = "test"
        test_pred["model_family"] = "baseline"
        test_predictions.append(test_pred)

    validation_frame_out = (
        pd.concat(validation_predictions, axis=0, ignore_index=True)
        if validation_predictions
        else pd.DataFrame()
    )
    test_frame_out = (
        pd.concat(test_predictions, axis=0, ignore_index=True) if test_predictions else pd.DataFrame()
    )

    return BaselinePredictionBundle(
        validation_predictions=validation_frame_out,
        test_predictions=test_frame_out,
    )
