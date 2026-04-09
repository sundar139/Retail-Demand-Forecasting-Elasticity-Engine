"""Tests for deterministic LLM feature planning and materialization."""

from pathlib import Path
from typing import Any, cast

import pandas as pd

from retail_forecasting.feature_spec import (
    BinaryFlagFeatureSpec,
    CalendarFeatureSpec,
    DifferenceFeatureSpec,
    LagFeatureSpec,
    RatioFeatureSpec,
    RollingFeatureSpec,
)
from retail_forecasting.features_llm import build_llm_features_pipeline, materialize_llm_features, plan_llm_features_pipeline
from retail_forecasting.ollama_client import OllamaPlannerResponse
from retail_forecasting.paths import ProjectPaths
from retail_forecasting.preprocessing import load_json


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


class _UnavailablePlanner:
    def plan_feature_specs(self, system_prompt: str, user_prompt: str) -> OllamaPlannerResponse:
        return OllamaPlannerResponse(
            reachable=False,
            host="http://localhost:11434",
            model="mock-model",
            raw_response_text="",
            parsed_json=None,
            error="connection refused",
        )


class _DuplicateNamePlanner:
    def plan_feature_specs(self, system_prompt: str, user_prompt: str) -> OllamaPlannerResponse:
        return OllamaPlannerResponse(
            reachable=True,
            host="http://localhost:11434",
            model="mock-model",
            raw_response_text='{"specs": []}',
            parsed_json={
                "specs": [
                    {
                        "operation": "lag_feature",
                        "feature_name": "units_lag_dup_llm",
                        "source_column": "units_sold",
                        "lag": 1,
                        "group_by": ["store_id", "product_id"],
                    },
                    {
                        "operation": "lag_feature",
                        "feature_name": "units_lag_dup_llm",
                        "source_column": "units_sold",
                        "lag": 2,
                        "group_by": ["store_id", "product_id"],
                    },
                ]
            },
            error=None,
        )


def _build_source_frame(num_days: int = 12) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=num_days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "store_id": ["S1"] * num_days,
            "product_id": ["P1"] * num_days,
            "units_sold": [10 + day for day in range(num_days)],
            "price": [5.0 + ((day % 3) * 0.1) for day in range(num_days)],
            "discount": [0.1 if day % 2 == 0 else 0.0 for day in range(num_days)],
            "demand_forecast": [11.0 + day for day in range(num_days)],
            "inventory_level": [100.0 - day for day in range(num_days)],
        }
    )


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

    (prompts / "feature_planner_system.txt").write_text(
        "System planner prompt for tests.",
        encoding="utf-8",
    )
    (prompts / "feature_planner_user_template.txt").write_text(
        "Columns {available_columns_json} schema {feature_plan_schema_json}",
        encoding="utf-8",
    )

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


def test_materialize_llm_features_supports_safe_operations() -> None:
    """Materializer should support the safe deterministic operation subset."""
    source = _build_source_frame(num_days=12)

    specs = [
        LagFeatureSpec(
            operation="lag_feature",
            feature_name="units_sold_lag_1_llm",
            source_column="units_sold",
            lag=1,
            group_by=["store_id", "product_id"],
        ),
        RollingFeatureSpec(
            operation="rolling_feature",
            feature_name="units_sold_roll_mean_3_llm",
            source_column="units_sold",
            window=3,
            aggregation="mean",
            shift=1,
            group_by=["store_id", "product_id"],
        ),
        RatioFeatureSpec(
            operation="ratio_feature",
            feature_name="price_to_inventory_llm",
            numerator_column="price",
            denominator_column="inventory_level",
        ),
        DifferenceFeatureSpec(
            operation="difference_feature",
            feature_name="price_minus_discount_llm",
            minuend_column="price",
            subtrahend_column="discount",
        ),
        BinaryFlagFeatureSpec(
            operation="binary_flag_feature",
            feature_name="discount_positive_flag_llm",
            source_column="discount",
            comparator="gt",
            threshold=0.0,
        ),
        CalendarFeatureSpec(
            operation="calendar_feature",
            feature_name="is_weekend_x_discount_llm",
            calendar_component="is_weekend",
            interact_with_column="discount",
        ),
    ]

    feature_frame, created, rejected = materialize_llm_features(source, specs)

    assert len(created) == 6
    assert not rejected
    assert "units_sold_lag_1_llm" in feature_frame.columns
    assert pd.isna(feature_frame.loc[0, "units_sold_lag_1_llm"])
    assert feature_frame.loc[1, "units_sold_lag_1_llm"] == feature_frame.loc[0, "units_sold"]


def test_rolling_materialization_uses_prior_only_history() -> None:
    """Rolling materialization must remain shift-based and leakage-safe."""
    source_a = _build_source_frame(num_days=14)
    source_b = source_a.copy(deep=True)
    source_b.loc[8, "units_sold"] = 9999.0

    spec = RollingFeatureSpec(
        operation="rolling_feature",
        feature_name="units_sold_roll_mean_3_llm",
        source_column="units_sold",
        window=3,
        aggregation="mean",
        shift=1,
        group_by=["store_id", "product_id"],
    )

    frame_a, _, _ = materialize_llm_features(source_a, [spec])
    frame_b, _, _ = materialize_llm_features(source_b, [spec])

    assert frame_a.loc[8, "units_sold_roll_mean_3_llm"] == frame_b.loc[8, "units_sold_roll_mean_3_llm"]
    assert frame_a.loc[9, "units_sold_roll_mean_3_llm"] != frame_b.loc[9, "units_sold_roll_mean_3_llm"]


def test_lag_materialization_uses_prior_only_history() -> None:
    """Lag materialization must use only historical values from prior rows."""
    source_a = _build_source_frame(num_days=12)
    source_b = source_a.copy(deep=True)
    source_b.loc[6, "units_sold"] = 7000.0

    spec = LagFeatureSpec(
        operation="lag_feature",
        feature_name="units_sold_lag_1_llm",
        source_column="units_sold",
        lag=1,
        group_by=["store_id", "product_id"],
    )

    frame_a, _, _ = materialize_llm_features(source_a, [spec])
    frame_b, _, _ = materialize_llm_features(source_b, [spec])

    assert frame_a.loc[6, "units_sold_lag_1_llm"] == frame_b.loc[6, "units_sold_lag_1_llm"]
    assert frame_a.loc[7, "units_sold_lag_1_llm"] != frame_b.loc[7, "units_sold_lag_1_llm"]


def test_materialization_rejects_duplicate_existing_feature_names() -> None:
    """Materializer should reject specs colliding with existing frame columns."""
    source = _build_source_frame(num_days=10)
    source["llm_collision_feature"] = 1.0

    spec = DifferenceFeatureSpec(
        operation="difference_feature",
        feature_name="llm_collision_feature",
        minuend_column="price",
        subtrahend_column="discount",
    )

    _, created, rejected = materialize_llm_features(source, [spec])

    assert not created
    assert len(rejected) == 1
    assert "collision" in str(rejected[0]["reason"])


def test_plan_pipeline_handles_ollama_unavailable_fallback(tmp_path: Path, monkeypatch) -> None:
    """Planning pipeline should gracefully write artifacts when Ollama is unavailable."""
    from retail_forecasting import features_llm

    paths = _build_temp_paths(tmp_path)
    source_path = paths.data_interim_dir / "cleaned_retail.parquet"
    _build_source_frame(num_days=8).to_parquet(source_path, index=False)

    monkeypatch.setattr(features_llm, "build_project_paths", lambda: paths)

    outputs = plan_llm_features_pipeline(
        input_path=source_path,
        planner=cast(Any, _UnavailablePlanner()),
    )
    summary = load_json(outputs["llm_features_summary_json"])

    assert outputs["llm_feature_plan_raw_json"].exists()
    assert outputs["llm_feature_plan_validated_json"].exists()
    assert outputs["llm_features_summary_json"].exists()
    assert summary["ollama_reachable"] is False
    assert _as_int(summary.get("accepted_spec_count", 0)) == 0
    assert _as_int(summary.get("output_feature_count", 0)) == 0
    assert _as_int(summary.get("rejected_spec_count", 0)) >= 1


def test_build_pipeline_rejects_duplicate_feature_names(tmp_path: Path, monkeypatch) -> None:
    """Duplicate feature names from planner should be rejected deterministically."""
    from retail_forecasting import features_llm

    paths = _build_temp_paths(tmp_path)
    source_path = paths.data_interim_dir / "cleaned_retail.parquet"
    _build_source_frame(num_days=9).to_parquet(source_path, index=False)

    monkeypatch.setattr(features_llm, "build_project_paths", lambda: paths)

    outputs = build_llm_features_pipeline(
        input_path=source_path,
        planner=cast(Any, _DuplicateNamePlanner()),
    )
    summary = load_json(outputs["llm_features_summary_json"])

    assert _as_int(summary.get("raw_spec_count", 0)) == 2
    assert _as_int(summary.get("accepted_spec_count", 0)) == 1
    assert _as_int(summary.get("rejected_spec_count", 0)) == 1
    assert outputs["features_llm_parquet"].exists()
