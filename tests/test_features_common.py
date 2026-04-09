"""Unit tests for reusable manual feature helper utilities."""

import numpy as np
import pandas as pd

from retail_forecasting.features_common import (
    add_group_lag_features,
    add_group_shifted_rolling_features,
    safe_ratio,
)


def test_group_lag_features_are_correct_and_group_isolated() -> None:
    """Lag values should be computed only from prior rows in the same group."""
    frame = pd.DataFrame(
        {
            "store_id": ["S1", "S1", "S2", "S2"],
            "product_id": ["P1", "P1", "P2", "P2"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "units_sold": [10.0, 11.0, 100.0, 101.0],
        }
    )

    generated = add_group_lag_features(frame, "units_sold", lags=(1,), feature_prefix="units_sold")

    assert generated == ["units_sold_lag_1"]
    assert pd.isna(frame.loc[0, "units_sold_lag_1"])
    assert frame.loc[1, "units_sold_lag_1"] == 10.0
    assert pd.isna(frame.loc[2, "units_sold_lag_1"])
    assert frame.loc[3, "units_sold_lag_1"] == 100.0


def test_shifted_rolling_feature_uses_only_prior_observations() -> None:
    """Shifted rolling stats must not include the current row value."""
    frame = pd.DataFrame(
        {
            "store_id": ["S1", "S1", "S1", "S1"],
            "product_id": ["P1", "P1", "P1", "P1"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "units_sold": [1.0, 2.0, 100.0, 4.0],
        }
    )

    generated = add_group_shifted_rolling_features(
        dataframe=frame,
        source_column="units_sold",
        window_to_statistics={2: ("mean",)},
        feature_prefix="units_sold",
        shift_periods=1,
    )

    assert generated == ["units_sold_roll_mean_2"]
    # For 2024-01-03 this should be mean(1, 2) and must not include current value 100.
    assert frame.loc[2, "units_sold_roll_mean_2"] == 1.5
    # For 2024-01-04 this should be mean(2, 100).
    assert frame.loc[3, "units_sold_roll_mean_2"] == 51.0


def test_safe_ratio_handles_zero_and_missing_denominators() -> None:
    """Safe ratio should output NaN instead of inf when denominator is invalid."""
    numerator = pd.Series([1.0, 2.0, 3.0], dtype="float64")
    denominator = pd.Series([1.0, 0.0, np.nan], dtype="float64")

    result = safe_ratio(numerator, denominator)

    assert result.iloc[0] == 1.0
    assert np.isnan(result.iloc[1])
    assert np.isnan(result.iloc[2])
    assert not np.isinf(result.to_numpy(dtype="float64", na_value=np.nan)).any()
