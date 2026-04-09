"""Tests for CSV discovery and loading."""

from pathlib import Path

import pandas as pd
import pytest

from retail_forecasting.data_loading import discover_and_load_csv, discover_raw_csv


def test_discover_raw_csv_single_candidate(tmp_path: Path) -> None:
    """A single CSV in data/raw should be selected automatically."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "retail.csv"
    csv_path.write_text("date,store_id,product_id,units_sold,price\n2024-01-01,S1,P1,10,9.9\n", encoding="utf-8")

    discovered = discover_raw_csv(raw_dir=raw_dir)
    assert discovered == csv_path


def test_discover_raw_csv_requires_explicit_when_multiple(tmp_path: Path) -> None:
    """Multiple CSV files should require explicit path selection."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "a.csv").write_text("x\n1\n", encoding="utf-8")
    (raw_dir / "b.csv").write_text("x\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Multiple CSV files"):
        discover_raw_csv(raw_dir=raw_dir)


def test_discover_and_load_csv_explicit_path(tmp_path: Path) -> None:
    """Explicit CSV paths should load into a dataframe."""
    csv_path = tmp_path / "manual.csv"
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "store": ["S1"],
            "sku": ["P1"],
            "sales": [5],
            "unit_price": [4.5],
        }
    )
    frame.to_csv(csv_path, index=False)

    discovered, loaded = discover_and_load_csv(input_path=csv_path)
    assert discovered == csv_path
    assert loaded.shape == (1, 5)
