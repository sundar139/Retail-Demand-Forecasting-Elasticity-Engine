"""Tests for causal utility helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retail_forecasting.causal_utils import (
    load_causal_feature_frame,
    safe_log_transform,
    select_causal_control_features,
    select_control_feature_columns,
)
from retail_forecasting.paths import ProjectPaths


def _build_temp_paths(tmp_path: Path) -> ProjectPaths:
    data_raw = tmp_path / "data" / "raw"
    data_interim = tmp_path / "data" / "interim"
    data_processed = tmp_path / "data" / "processed"
    artifacts = tmp_path / "artifacts"
    notebooks = tmp_path / "notebooks"
    reports = tmp_path / "reports" / "figures"
    prompts = tmp_path / "prompts"
    scripts = tmp_path / "scripts"

    for directory in [
        data_raw,
        data_interim,
        data_processed,
        artifacts,
        notebooks,
        reports,
        prompts,
        scripts,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        project_root=tmp_path,
        data_raw_dir=data_raw,
        data_interim_dir=data_interim,
        data_processed_dir=data_processed,
        artifacts_dir=artifacts,
        notebooks_dir=notebooks,
        reports_figures_dir=reports,
        prompts_dir=prompts,
        scripts_dir=scripts,
    )


def _manual_feature_frame(days: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "store_id": ["S1"] * days,
            "product_id": ["P1"] * days,
            "units_sold": [20 + day for day in range(days)],
            "price": [10.0 + ((day % 4) * 0.2) for day in range(days)],
            "discount": [0.1 if day % 2 == 0 else 0.0 for day in range(days)],
            "demand_forecast": [22 + day for day in range(days)],
            "inventory_level": [120 - day for day in range(days)],
            "units_sold_lag_1": [np.nan] + [20 + day for day in range(days - 1)],
        }
    )


def test_safe_log_transform_handles_zero_with_positive_epsilon() -> None:
    """Safe log transform should stay finite for zero values with positive epsilon."""
    series = pd.Series([0.0, 1.0, 9.0], dtype="float64")
    transformed = safe_log_transform(series, epsilon=1e-3)

    assert np.isfinite(transformed.to_numpy()).all()
    assert transformed.iloc[0] < transformed.iloc[1] < transformed.iloc[2]


def test_safe_log_transform_rejects_invalid_negative_values() -> None:
    """Values below -epsilon should be rejected as unsafe for log transform."""
    series = pd.Series([-1.0, 0.0, 1.0], dtype="float64")

    with pytest.raises(ValueError, match="log undefined"):
        safe_log_transform(series, epsilon=1e-3)


def test_select_control_feature_columns_excludes_forbidden_columns() -> None:
    """Control selection should exclude identifiers, treatment, target, and leakage columns."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "store_id": ["S1", "S1", "S1"],
            "product_id": ["P1", "P1", "P1"],
            "units_sold": [10, 12, 9],
            "price": [5.0, 5.2, 5.1],
            "forecast_error_hint": [0.1, 0.0, -0.1],
            "units_sold_lag_1": [np.nan, 10.0, 12.0],
            "discount": [0.0, 0.1, 0.0],
            "text_col": ["a", "b", "c"],
        }
    )

    selected = select_control_feature_columns(frame)

    assert "units_sold_lag_1" in selected
    assert "discount" in selected
    assert "units_sold" not in selected
    assert "price" not in selected
    assert "forecast_error_hint" not in selected
    assert "store_id" not in selected
    assert "text_col" not in selected


def test_select_causal_control_features_lean_excludes_own_price_proxies() -> None:
    """Lean causal profile should exclude own-price lag/rolling/momentum proxy controls."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "store_id": ["S1", "S1", "S1", "S1"],
            "product_id": ["P1", "P1", "P1", "P1"],
            "price": [9.0, 9.2, 9.1, 9.3],
            "price_lag_1": [8.8, 9.0, 9.2, 9.1],
            "price_roll_mean_7": [8.9, 9.0, 9.1, 9.15],
            "price_momentum_7": [0.0, 0.01, -0.01, 0.02],
            "day_of_week": [1, 2, 3, 4],
            "month": [1, 1, 1, 1],
            "is_weekend": [False, False, False, False],
            "units_sold_lag_7": [12.0, 13.0, 14.0, 15.0],
            "units_sold_lag_28": [9.0, 9.0, 10.0, 10.0],
            "discount": [0.0, 0.1, 0.0, 0.1],
            "inventory_level": [100.0, 98.0, 97.0, 96.0],
            "demand_forecast": [20.0, 21.0, 22.0, 23.0],
        }
    )

    selected = select_causal_control_features(frame, feature_profile="lean")

    assert "price_lag_1" not in selected
    assert "price_roll_mean_7" not in selected
    assert "price_momentum_7" not in selected
    assert "demand_forecast" not in selected
    assert "units_sold_lag_7" in selected
    assert "units_sold_lag_28" in selected
    assert "discount" in selected
    assert "inventory_level" in selected


def test_select_causal_control_features_lean_excludes_llm_features_by_default() -> None:
    """Lean profile should not include llm-derived columns unless explicitly enabled."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "store_id": ["S1", "S1", "S1"],
            "product_id": ["P1", "P1", "P1"],
            "discount": [0.0, 0.1, 0.2],
            "units_sold_lag_7": [5.0, 6.0, 7.0],
            "llm_signal": [0.3, 0.5, 0.7],
            "promo_llm_score": [0.4, 0.6, 0.8],
        }
    )

    selected = select_causal_control_features(frame, feature_profile="lean")

    assert "llm_signal" not in selected
    assert "promo_llm_score" not in selected
    assert "discount" in selected
    assert "units_sold_lag_7" in selected


def test_feature_profile_lean_selects_fewer_controls_than_full() -> None:
    """Lean profile should yield a smaller, auditable control set than full profile."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "store_id": ["S1"] * 6,
            "product_id": ["P1"] * 6,
            "price": [8.0, 8.2, 8.1, 8.3, 8.4, 8.5],
            "day_of_week": [0, 1, 2, 3, 4, 5],
            "month": [1] * 6,
            "is_weekend": [False, False, False, False, False, True],
            "units_sold_lag_7": [9.0, 9.5, 10.0, 10.5, 11.0, 11.5],
            "units_sold_lag_28": [7.0, 7.0, 7.5, 7.5, 8.0, 8.0],
            "units_sold_lag_1": [10.0, 10.5, 11.0, 11.5, 12.0, 12.5],
            "discount": [0.0, 0.1, 0.0, 0.05, 0.1, 0.0],
            "inventory_level": [120.0, 119.0, 118.0, 117.0, 116.0, 115.0],
            "demand_forecast": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "random_numeric_feature": [1.0, 2.0, 1.5, 2.5, 2.0, 3.0],
            "price_lag_1": [7.9, 8.0, 8.2, 8.1, 8.3, 8.4],
        }
    )

    lean_selected = select_causal_control_features(frame, feature_profile="lean")
    full_selected = select_causal_control_features(frame, feature_profile="full")

    assert len(lean_selected) < len(full_selected)
    assert "random_numeric_feature" not in lean_selected
    assert "random_numeric_feature" in full_selected
    assert "price_lag_1" not in lean_selected
    assert "price_lag_1" not in full_selected


def test_load_causal_feature_frame_handles_missing_optional_llm_file(tmp_path: Path) -> None:
    """Loading should continue with manual features when optional LLM file is absent."""
    paths = _build_temp_paths(tmp_path)
    manual_path = paths.data_processed_dir / "features_manual.parquet"
    _manual_feature_frame().to_parquet(manual_path, index=False)

    loaded = load_causal_feature_frame(project_paths=paths, use_llm_features=True)

    assert loaded.source_data_path == manual_path
    assert loaded.llm_augmentation_used is False
    assert not loaded.llm_added_columns
    assert "units_sold_lag_1" in loaded.frame.columns


def test_load_causal_feature_frame_handles_empty_llm_features(tmp_path: Path) -> None:
    """Empty LLM features should not break loading and should be skipped cleanly."""
    paths = _build_temp_paths(tmp_path)
    manual_path = paths.data_processed_dir / "features_manual.parquet"
    _manual_feature_frame().to_parquet(manual_path, index=False)

    llm_path = paths.data_processed_dir / "features_llm.parquet"
    pd.DataFrame(columns=["date", "store_id", "product_id"]).to_parquet(llm_path, index=False)

    loaded = load_causal_feature_frame(project_paths=paths, use_llm_features=True)

    assert loaded.llm_augmentation_used is False
    assert not loaded.llm_added_columns
    assert any("empty" in note for note in loaded.notes)


def test_load_causal_feature_frame_merges_non_overlapping_llm_columns(tmp_path: Path) -> None:
    """Non-overlapping LLM columns should be merged as optional augmentation."""
    paths = _build_temp_paths(tmp_path)
    manual_frame = _manual_feature_frame()
    manual_path = paths.data_processed_dir / "features_manual.parquet"
    manual_frame.to_parquet(manual_path, index=False)

    llm_frame = manual_frame[["date", "store_id", "product_id"]].copy(deep=True)
    llm_frame["llm_demand_signal"] = np.linspace(0.1, 0.9, len(llm_frame))
    llm_path = paths.data_processed_dir / "features_llm.parquet"
    llm_frame.to_parquet(llm_path, index=False)

    loaded = load_causal_feature_frame(project_paths=paths, use_llm_features=True)

    assert loaded.llm_augmentation_used is True
    assert loaded.llm_added_columns == ["llm_demand_signal"]
    assert "llm_demand_signal" in loaded.frame.columns
