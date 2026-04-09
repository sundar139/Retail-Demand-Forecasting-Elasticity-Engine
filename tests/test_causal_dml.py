"""Tests for segmented causal DML elasticity pipeline."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from retail_forecasting.causal_dml import (
    ElasticityRunConfig,
    fit_elasticity_pipeline,
    load_elasticity_estimates,
    load_elasticity_run_summary,
)
from retail_forecasting.paths import ProjectPaths


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Value cannot be coerced to int: {value!r}")


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


def _synthetic_manual_features() -> pd.DataFrame:
    rng = np.random.default_rng(42)

    rows: list[dict[str, object]] = []

    # Segment expected to pass and exhibit negative own-price elasticity.
    for day in range(120):
        date_value = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
        price = float(8.0 + rng.normal(0.0, 0.4))
        discount = float(max(0.0, rng.normal(0.08, 0.03)))
        inventory_level = float(150 + rng.normal(0.0, 8.0))
        demand_forecast = float(22 + rng.normal(0.0, 1.2))

        latent_log_units = (
            3.2
            - 1.35 * np.log(price + 1e-3)
            + 0.25 * discount
            + 0.01 * (inventory_level / 10.0)
            + 0.02 * demand_forecast
            + rng.normal(0.0, 0.08)
        )
        units_sold = float(max(np.exp(latent_log_units) - 1e-3, 0.0))

        rows.append(
            {
                "date": date_value,
                "store_id": "S1",
                "product_id": "P_GOOD",
                "units_sold": units_sold,
                "price": price,
                "discount": discount,
                "demand_forecast": demand_forecast,
                "inventory_level": inventory_level,
                "units_sold_lag_1": np.nan if day == 0 else rows[-1]["units_sold"],
                "price_roll_mean_7": price,
            }
        )

    # Segment expected to fail guardrails due to low sample size and low variation.
    for day in range(20):
        date_value = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
        rows.append(
            {
                "date": date_value,
                "store_id": "S2",
                "product_id": "P_BAD",
                "units_sold": 12.0,
                "price": 6.0,
                "discount": 0.0,
                "demand_forecast": 12.0,
                "inventory_level": 80.0,
                "units_sold_lag_1": 12.0,
                "price_roll_mean_7": 6.0,
            }
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(["store_id", "product_id", "date"], kind="mergesort").reset_index(drop=True)


def test_fit_pipeline_handles_mixed_segment_outcomes(tmp_path: Path, monkeypatch) -> None:
    """Pipeline should continue when some segments fit and others are skipped."""
    from retail_forecasting import causal_dml

    paths = _build_temp_paths(tmp_path)
    monkeypatch.setattr(causal_dml, "build_project_paths", lambda: paths)

    manual_path = paths.data_processed_dir / "features_manual.parquet"
    _synthetic_manual_features().to_parquet(manual_path, index=False)

    outputs = fit_elasticity_pipeline(
        ElasticityRunConfig(
            segment_level="product",
            min_samples=40,
            epsilon=1e-3,
            use_llm_features=False,
            nuisance_model="random-forest",
        )
    )

    assert outputs["elasticity_estimates_csv"].exists()
    assert outputs["elasticity_run_summary_json"].exists()

    estimates = load_elasticity_estimates(outputs["elasticity_estimates_csv"])
    summary = load_elasticity_run_summary(outputs["elasticity_run_summary_json"])

    assert _as_int(summary.get("total_segments_attempted", 0)) == 2
    assert _as_int(summary.get("successful_fits", 0)) >= 1
    assert _as_int(summary.get("skipped_fits", 0)) >= 1
    assert set(estimates["fit_status"].astype("string").tolist()) & {"success", "skipped"}


def test_fit_pipeline_recovers_negative_direction_on_synthetic_data(tmp_path: Path, monkeypatch) -> None:
    """Estimated elasticity direction should be negative on synthetic negative-price-effect data."""
    from retail_forecasting import causal_dml

    paths = _build_temp_paths(tmp_path)
    monkeypatch.setattr(causal_dml, "build_project_paths", lambda: paths)

    manual_path = paths.data_processed_dir / "features_manual.parquet"
    synthetic = _synthetic_manual_features()
    synthetic = synthetic.loc[synthetic["product_id"] == "P_GOOD"].reset_index(drop=True)
    synthetic.to_parquet(manual_path, index=False)

    outputs = fit_elasticity_pipeline(
        ElasticityRunConfig(
            segment_level="product",
            min_samples=50,
            epsilon=1e-3,
            use_llm_features=False,
            nuisance_model="gradient-boosting",
        )
    )

    estimates = load_elasticity_estimates(outputs["elasticity_estimates_csv"])
    successful = estimates.loc[estimates["fit_status"] == "success"].copy()

    assert not successful.empty
    mean_estimate = float(successful["elasticity_estimate"].mean())
    assert mean_estimate < 0.0


def test_pipeline_handles_missing_or_empty_llm_features_gracefully(tmp_path: Path, monkeypatch) -> None:
    """Optional LLM features should not be required for successful pipeline execution."""
    from retail_forecasting import causal_dml

    paths = _build_temp_paths(tmp_path)
    monkeypatch.setattr(causal_dml, "build_project_paths", lambda: paths)

    manual_path = paths.data_processed_dir / "features_manual.parquet"
    _synthetic_manual_features().to_parquet(manual_path, index=False)

    # Case 1: missing LLM features file.
    outputs_missing = fit_elasticity_pipeline(
        ElasticityRunConfig(
            segment_level="product",
            min_samples=40,
            use_llm_features=True,
        )
    )
    summary_missing = load_elasticity_run_summary(outputs_missing["elasticity_run_summary_json"])
    assert summary_missing["llm_feature_augmentation_used"] is False

    # Case 2: present but empty/non-overlapping LLM file.
    llm_path = paths.data_processed_dir / "features_llm.parquet"
    pd.DataFrame(columns=["date", "store_id", "product_id"]).to_parquet(llm_path, index=False)

    outputs_empty = fit_elasticity_pipeline(
        ElasticityRunConfig(
            segment_level="product",
            min_samples=40,
            use_llm_features=True,
        )
    )
    summary_empty = load_elasticity_run_summary(outputs_empty["elasticity_run_summary_json"])
    assert summary_empty["llm_feature_augmentation_used"] is False


def test_inference_warning_is_propagated_to_quality_and_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Inference warning text should force warning quality status and run-level warning flags."""
    from retail_forecasting import causal_dml

    class _WarningLinearDML:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        def fit(self, Y, T, X) -> None:
            _ = Y, T, X
            warnings.warn(
                "Co-variance matrix is underdetermined. Inference will be invalid!",
                RuntimeWarning,
            )

        def const_marginal_effect(self, X):
            return np.full((len(X), 1), -1.1)

        def const_marginal_effect_interval(self, X, alpha=0.05):
            _ = alpha
            lower = np.full((len(X), 1), -1.6)
            upper = np.full((len(X), 1), -0.7)
            return lower, upper

    paths = _build_temp_paths(tmp_path)
    monkeypatch.setattr(causal_dml, "build_project_paths", lambda: paths)
    monkeypatch.setattr(causal_dml, "LinearDML", _WarningLinearDML)

    manual_path = paths.data_processed_dir / "features_manual.parquet"
    synthetic = _synthetic_manual_features()
    synthetic = synthetic.loc[synthetic["product_id"] == "P_GOOD"].reset_index(drop=True)
    synthetic.to_parquet(manual_path, index=False)

    outputs = fit_elasticity_pipeline(
        ElasticityRunConfig(
            segment_level="product",
            min_samples=40,
            use_llm_features=False,
        )
    )

    estimates = load_elasticity_estimates(outputs["elasticity_estimates_csv"])
    summary = load_elasticity_run_summary(outputs["elasticity_run_summary_json"])

    assert (estimates["quality_status"].astype("string") == "warning_inference_unstable").any()
    assert (pd.to_numeric(estimates["inference_warning_count"], errors="coerce").fillna(0) > 0).any()
    assert bool(summary.get("inference_warnings_present", False)) is True
    assert _as_int(summary.get("inference_warning_count", 0)) > 0
    assert bool(summary.get("ci_caution_present", False)) is True

    quality_counts = summary.get("quality_status_counts", {})
    assert isinstance(quality_counts, dict)
    assert _as_int(quality_counts.get("warning_inference_unstable", 0)) >= 1


def test_low_variation_price_segment_is_skipped_by_guardrails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Segments with insufficient unique/variable prices should be skipped with explicit reasons."""
    from retail_forecasting import causal_dml

    paths = _build_temp_paths(tmp_path)
    monkeypatch.setattr(causal_dml, "build_project_paths", lambda: paths)

    dates = pd.date_range("2024-01-01", periods=320, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "store_id": ["S1"] * len(dates),
            "product_id": ["P_LOW_VAR"] * len(dates),
            "units_sold": np.linspace(20.0, 30.0, len(dates)),
            "price": [7.5] * len(dates),
            "discount": np.where(np.arange(len(dates)) % 3 == 0, 0.1, 0.0),
            "inventory_level": np.linspace(140.0, 120.0, len(dates)),
            "units_sold_lag_7": np.linspace(18.0, 28.0, len(dates)),
            "units_sold_lag_28": np.linspace(16.0, 26.0, len(dates)),
            "day_of_week": [int(day.dayofweek) for day in dates],
            "month": [int(day.month) for day in dates],
            "is_weekend": [bool(day.dayofweek >= 5) for day in dates],
        }
    )

    frame.to_parquet(paths.data_processed_dir / "features_manual.parquet", index=False)

    outputs = fit_elasticity_pipeline(
        ElasticityRunConfig(
            segment_level="product",
            feature_profile="lean",
            min_samples=250,
            min_non_null_pairs=250,
            min_unique_price_values=8,
            min_log_price_std=0.01,
            min_log_units_std=0.01,
            use_llm_features=False,
        )
    )

    estimates = load_elasticity_estimates(outputs["elasticity_estimates_csv"])
    assert not estimates.empty
    assert (estimates["fit_status"].astype("string") == "skipped").all()

    skip_reasons = estimates["skip_reason"].fillna("").astype("string").str.lower().tolist()
    assert any("unique_price_values_below_threshold" in reason for reason in skip_reasons)
    assert "unique_price_values" in estimates.columns
    assert "log_price_std" in estimates.columns
