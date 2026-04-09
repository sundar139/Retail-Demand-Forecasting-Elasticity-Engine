"""Tests for reusable forecasting baselines."""

import numpy as np
import pandas as pd

from retail_forecasting.baselines import (
    BASELINE_SEASONAL_NAIVE_7,
    DEFAULT_BASELINE_MODELS,
    generate_baseline_predictions,
    run_baseline_suite,
)


def _build_split_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    rows: list[dict[str, object]] = []

    for day_index, date_value in enumerate(dates, start=1):
        rows.append(
            {
                "date": date_value,
                "store_id": "S1",
                "product_id": "P1",
                "units_sold": float(day_index),
                "price": 10.0,
                "units_sold_lag_1": np.nan if day_index == 1 else float(day_index - 1),
                "units_sold_roll_mean_7": float(max(day_index - 1, 1)),
            }
        )

    frame = pd.DataFrame(rows)
    train = frame.iloc[:20].reset_index(drop=True)
    validation = frame.iloc[20:25].reset_index(drop=True)
    test = frame.iloc[25:].reset_index(drop=True)
    return train, validation, test


def test_baseline_prediction_alignment_matches_target_rows() -> None:
    """All baseline predictions should align one-to-one with split rows."""
    train, validation, test = _build_split_frames()

    bundle = run_baseline_suite(train, validation, test)

    assert len(bundle.validation_predictions) == len(validation) * len(DEFAULT_BASELINE_MODELS)
    assert len(bundle.test_predictions) == len(test) * len(DEFAULT_BASELINE_MODELS)

    assert bundle.validation_predictions["prediction"].notna().all()
    assert bundle.test_predictions["prediction"].notna().all()


def test_seasonal_naive_uses_week_lag_when_history_available() -> None:
    """Seasonal-naive baseline should use the same weekday value from 7 days earlier."""
    train, validation, _ = _build_split_frames()

    predictions = generate_baseline_predictions(
        history_frame=train,
        target_frame=validation,
        baseline_name=BASELINE_SEASONAL_NAIVE_7,
    )

    # Validation starts on day 21, so seasonal-7 prediction should come from day 14.
    assert float(predictions.iloc[0]["prediction"]) == 14.0
    assert predictions["prediction"].notna().all()
