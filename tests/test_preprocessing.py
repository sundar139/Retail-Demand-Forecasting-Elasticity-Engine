"""Tests for cleaning and deterministic chronological split behavior."""

import pandas as pd

from retail_forecasting.preprocessing import clean_retail_dataframe, split_chronologically


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


def _synthetic_raw_frame() -> pd.DataFrame:
    """Create synthetic raw data with aliases and duplicate rows."""
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    rows: list[dict[str, object]] = []

    for date_value in dates:
        rows.append(
            {
                "sale_date": str(date_value.date()),
                "store": "S1",
                "sku": "P1",
                "sales": 10,
                "unit_price": 4.0,
                "promotion": "yes" if date_value.day % 2 == 0 else "no",
            }
        )

    # Add an exact duplicate row.
    rows.append(rows[-1].copy())
    return pd.DataFrame(rows)


def test_clean_retail_dataframe_removes_duplicates_and_sorts() -> None:
    """Cleaning should remove exact duplicates and produce deterministic ordering."""
    raw = _synthetic_raw_frame()
    cleaned, summary = clean_retail_dataframe(raw, source_filename="retail.csv")

    assert _as_int(summary.get("duplicate_row_count_removed", 0)) == 1
    assert len(cleaned) == 12
    assert cleaned.equals(
        cleaned.sort_values(by=["store_id", "product_id", "date"], kind="mergesort").reset_index(drop=True)
    )


def test_chronological_split_correctness_and_non_overlap() -> None:
    """Splits must be chronological and non-overlapping by date."""
    raw = _synthetic_raw_frame()
    cleaned, _ = clean_retail_dataframe(raw, source_filename="retail.csv")

    splits, split_summary = split_chronologically(cleaned, train_ratio=0.70, validation_ratio=0.15, test_ratio=0.15)

    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    assert len(train) > 0
    assert len(validation) > 0
    assert len(test) > 0

    assert train["date"].max() < validation["date"].min()
    assert validation["date"].max() < test["date"].min()

    splits_payload = split_summary.get("splits")
    assert isinstance(splits_payload, dict)

    train_payload = splits_payload.get("train")
    validation_payload = splits_payload.get("validation")
    test_payload = splits_payload.get("test")

    assert isinstance(train_payload, dict)
    assert isinstance(validation_payload, dict)
    assert isinstance(test_payload, dict)

    assert _as_int(train_payload.get("row_count", 0)) == len(train)
    assert _as_int(validation_payload.get("row_count", 0)) == len(validation)
    assert _as_int(test_payload.get("row_count", 0)) == len(test)


def test_split_with_explicit_cutoff_dates() -> None:
    """Explicit cutoff dates should be honored exactly."""
    raw = _synthetic_raw_frame()
    cleaned, _ = clean_retail_dataframe(raw, source_filename="retail.csv")

    splits, _ = split_chronologically(
        cleaned,
        validation_start="2024-01-08",
        test_start="2024-01-10",
    )

    assert splits["train"]["date"].max() < pd.Timestamp("2024-01-08")
    assert splits["validation"]["date"].min() >= pd.Timestamp("2024-01-08")
    assert splits["validation"]["date"].max() < pd.Timestamp("2024-01-10")
    assert splits["test"]["date"].min() >= pd.Timestamp("2024-01-10")
