"""Unit tests for leakage-safe manual feature engineering."""

import numpy as np
import pandas as pd
import pytest

from retail_forecasting.features_manual import expected_manual_feature_names, generate_manual_features_frame


def _build_clean_frame(
    num_days: int = 40,
    groups: tuple[tuple[str, str], ...] = (("S1", "P1"),),
    include_optional_core: bool = True,
    force_zero_price: bool = False,
) -> pd.DataFrame:
    """Build synthetic cleaned data at store-product-date grain for tests."""
    dates = pd.date_range("2024-01-01", periods=num_days, freq="D")
    rows: list[dict[str, object]] = []

    for group_index, (store_id, product_id) in enumerate(groups):
        unit_offset = 20.0 + (group_index * 100.0)
        price_offset = 10.0 + (group_index * 2.0)

        for day_index, date_value in enumerate(dates):
            price = 0.0 if force_zero_price else price_offset + ((day_index % 5) * 0.1)
            row: dict[str, object] = {
                "date": date_value,
                "store_id": store_id,
                "product_id": product_id,
                "units_sold": unit_offset + day_index,
                "price": price,
            }

            if include_optional_core:
                row["discount"] = float((day_index % 4) * 0.05)
                row["demand_forecast"] = float(unit_offset + day_index + 1.0)
                row["inventory_level"] = float(max(0.0, 150.0 - day_index))

            rows.append(row)

    return pd.DataFrame(rows)


def test_stable_core_feature_names_are_generated() -> None:
    """The mandatory manual feature set should remain explicit and stable."""
    frame = _build_clean_frame(num_days=40, include_optional_core=False)

    feature_frame, metadata = generate_manual_features_frame(frame, drop_warmup_rows=False)
    generated_features = metadata.get("generated_features")

    assert isinstance(generated_features, list)

    expected = set(expected_manual_feature_names())
    assert expected.issubset(set(feature_frame.columns))
    assert expected.issubset(set(generated_features))


def test_warmup_trimming_drops_only_early_history_rows() -> None:
    """Warmup trimming should remove rows without full 28-step lag/rolling history."""
    frame = _build_clean_frame(num_days=40, groups=(("S1", "P1"),), include_optional_core=False)

    feature_frame, metadata = generate_manual_features_frame(frame, drop_warmup_rows=True)
    rows_dropped_for_warmup = metadata.get("rows_dropped_for_warmup")

    assert isinstance(rows_dropped_for_warmup, int)

    assert len(feature_frame) == 12
    assert rows_dropped_for_warmup == 28
    assert feature_frame["date"].min() == pd.Timestamp("2024-01-29")


def test_shifted_rolling_target_feature_is_leakage_safe() -> None:
    """Changing same-day target should not change same-day shifted rolling values."""
    frame_a = _build_clean_frame(num_days=40, include_optional_core=False)
    frame_b = frame_a.copy(deep=True)

    frame_b.loc[30, "units_sold"] = 5000.0

    features_a, _ = generate_manual_features_frame(frame_a, drop_warmup_rows=False)
    features_b, _ = generate_manual_features_frame(frame_b, drop_warmup_rows=False)

    target_date = pd.Timestamp(str(frame_a.loc[30, "date"]))
    next_date = pd.Timestamp(str(frame_a.loc[31, "date"]))

    row_a_target = features_a.loc[features_a["date"] == target_date].iloc[0]
    row_b_target = features_b.loc[features_b["date"] == target_date].iloc[0]
    row_a_next = features_a.loc[features_a["date"] == next_date].iloc[0]
    row_b_next = features_b.loc[features_b["date"] == next_date].iloc[0]

    assert row_a_target["units_sold"] != row_b_target["units_sold"]
    assert row_a_target["units_sold_roll_mean_7"] == pytest.approx(row_b_target["units_sold_roll_mean_7"])
    assert row_a_target["units_sold_roll_std_7"] == pytest.approx(row_b_target["units_sold_roll_std_7"])
    assert row_a_next["units_sold_roll_mean_7"] != row_b_next["units_sold_roll_mean_7"]


def test_no_cross_group_contamination_in_lags() -> None:
    """Lag features should never borrow history from different store-product groups."""
    frame = _build_clean_frame(
        num_days=35,
        groups=(("S1", "P1"), ("S2", "P2")),
        include_optional_core=False,
    )

    feature_frame, _ = generate_manual_features_frame(frame, drop_warmup_rows=False)

    grouped = feature_frame.groupby(["store_id", "product_id"], sort=True)
    for _, group_frame in grouped:
        ordered_group = group_frame.sort_values("date", kind="mergesort").reset_index(drop=True)
        assert pd.isna(ordered_group.loc[0, "units_sold_lag_1"])
        assert ordered_group.loc[1, "units_sold_lag_1"] == ordered_group.loc[0, "units_sold"]


def test_optional_columns_absent_are_handled_cleanly() -> None:
    """Optional feature groups should be skipped without breaking core features."""
    frame = _build_clean_frame(num_days=35, include_optional_core=False)

    feature_frame, metadata = generate_manual_features_frame(frame, drop_warmup_rows=False)
    skipped_missing = metadata.get("columns_skipped_missing")

    assert isinstance(skipped_missing, list)

    assert "discount" not in feature_frame.columns
    assert "demand_forecast" not in feature_frame.columns
    assert "inventory_level" not in feature_frame.columns

    skipped = set(skipped_missing)
    assert {"discount", "demand_forecast", "inventory_level"}.issubset(skipped)


def test_price_momentum_is_safe_with_zero_denominator_history() -> None:
    """Price momentum should produce NaN instead of inf when denominator history is zero."""
    frame = _build_clean_frame(num_days=35, include_optional_core=False, force_zero_price=True)

    feature_frame, _ = generate_manual_features_frame(frame, drop_warmup_rows=False)

    for column in ("price_momentum_7", "price_momentum_28"):
        values = feature_frame[column].astype("float64").to_numpy()
        assert not np.isinf(values).any()
        assert np.isnan(values).any()
