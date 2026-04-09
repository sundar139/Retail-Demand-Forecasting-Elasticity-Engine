"""Tests for structured LLM feature specification validation."""

import pandas as pd
import pytest

from retail_forecasting.feature_spec import (
    FeaturePlan,
    LagFeatureSpec,
    extract_raw_specs_from_payload,
    parse_feature_spec,
    validate_feature_plan_specs,
)


def test_valid_lag_spec_parses() -> None:
    """A valid lag spec should parse into the expected model type."""
    raw_spec = {
        "operation": "lag_feature",
        "feature_name": "units_sold_lag_3_llm",
        "source_column": "units_sold",
        "lag": 3,
        "group_by": ["store_id", "product_id"],
        "required_columns": ["units_sold", "date"],
    }

    parsed = parse_feature_spec(raw_spec)

    assert isinstance(parsed, LagFeatureSpec)
    assert parsed.feature_name == "units_sold_lag_3_llm"
    assert parsed.lag == 3


def test_unsupported_operation_is_rejected() -> None:
    """Unknown operation discriminators should fail schema validation."""
    raw_spec = {
        "operation": "custom_python_feature",
        "feature_name": "unsafe_feature",
    }

    with pytest.raises(Exception, match="operation"):
        parse_feature_spec(raw_spec)


def test_invalid_lag_and_reserved_name_are_rejected() -> None:
    """Invalid lag values and reserved feature names should be rejected."""
    negative_lag = {
        "operation": "lag_feature",
        "feature_name": "llm_feature_bad_lag",
        "source_column": "units_sold",
        "lag": -1,
        "group_by": ["store_id", "product_id"],
    }
    reserved_name = {
        "operation": "lag_feature",
        "feature_name": "price",
        "source_column": "units_sold",
        "lag": 1,
        "group_by": ["store_id", "product_id"],
    }

    with pytest.raises(Exception, match="lag"):
        parse_feature_spec(negative_lag)

    with pytest.raises(Exception, match="reserved"):
        parse_feature_spec(reserved_name)


def test_feature_plan_validation_rejects_duplicates_and_missing_columns() -> None:
    """Plan validation should reject duplicate names and missing required columns."""
    raw_specs = [
        {
            "operation": "lag_feature",
            "feature_name": "units_sold_lag_2_llm",
            "source_column": "units_sold",
            "lag": 2,
            "group_by": ["store_id", "product_id"],
        },
        {
            "operation": "lag_feature",
            "feature_name": "units_sold_lag_2_llm",
            "source_column": "units_sold",
            "lag": 3,
            "group_by": ["store_id", "product_id"],
        },
        {
            "operation": "ratio_feature",
            "feature_name": "price_to_competitor_llm",
            "numerator_column": "price",
            "denominator_column": "competitor_price",
        },
    ]

    available_columns = ["date", "store_id", "product_id", "units_sold", "price"]
    accepted, rejected = validate_feature_plan_specs(
        raw_specs=raw_specs,
        available_columns=available_columns,
        existing_feature_names=available_columns,
        blocked_feature_names=[],
    )

    assert len(accepted) == 1
    assert accepted[0].feature_name == "units_sold_lag_2_llm"
    assert len(rejected) == 2
    reasons = " | ".join(str(item["reason"]) for item in rejected)
    assert "duplicates" in reasons or "duplicate" in reasons
    assert "Missing required columns" in reasons


def test_feature_plan_accepts_features_key_alias() -> None:
    """Top-level plan should accept legacy features key and normalize to specs."""
    payload = {
        "plan_version": "1.0",
        "features": [
            {
                "operation": "lag_feature",
                "feature_name": "units_sold_lag_4_llm",
                "source_column": "units_sold",
                "lag": 4,
                "group_by": ["store_id", "product_id"],
            }
        ],
    }

    plan = FeaturePlan.model_validate(payload)
    extracted = extract_raw_specs_from_payload(payload)

    assert len(plan.specs) == 1
    assert plan.specs[0].feature_name == "units_sold_lag_4_llm"
    assert len(extracted) == 1


def test_validation_rejects_manual_namespace_overlap() -> None:
    """Feature names that overlap manual namespace should be rejected."""
    raw_specs = [
        {
            "operation": "difference_feature",
            "feature_name": "manual_overlap_llm",
            "minuend_column": "price",
            "subtrahend_column": "discount",
        }
    ]

    available_columns = [
        "date",
        "store_id",
        "product_id",
        "units_sold",
        "price",
        "discount",
    ]
    accepted, rejected = validate_feature_plan_specs(
        raw_specs=raw_specs,
        available_columns=available_columns,
        existing_feature_names=available_columns,
        blocked_feature_names=["manual_overlap_llm"],
    )

    assert not accepted
    assert len(rejected) == 1
    assert "manual feature namespace" in str(rejected[0]["reason"])


def test_validation_on_realistic_columns_snapshot() -> None:
    """Validation should pass for practical current dataset column availability."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "store_id": ["S1", "S1", "S1"],
            "product_id": ["P1", "P1", "P1"],
            "units_sold": [10, 12, 8],
            "price": [5.0, 5.2, 5.1],
            "discount": [0.1, 0.0, 0.2],
            "demand_forecast": [11, 12, 9],
            "inventory_level": [100, 95, 90],
        }
    )

    raw_specs = [
        {
            "operation": "ratio_feature",
            "feature_name": "price_to_inventory_llm",
            "numerator_column": "price",
            "denominator_column": "inventory_level",
        }
    ]

    accepted, rejected = validate_feature_plan_specs(
        raw_specs=raw_specs,
        available_columns=frame.columns.tolist(),
        existing_feature_names=frame.columns.tolist(),
        blocked_feature_names=[],
    )

    assert len(accepted) == 1
    assert not rejected
