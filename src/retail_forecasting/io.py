"""Input and output helpers for retail forecasting pipelines."""

from pathlib import Path
import logging

import pandas as pd

from retail_forecasting.schemas import DATE_COLUMN

LOGGER = logging.getLogger(__name__)


def _read_tabular_file(path: Path) -> pd.DataFrame:
    """Read tabular data from CSV or Parquet based on file extension.

    Args:
        path: Source file path.

    Returns:
        Loaded dataframe.

    Raises:
        ValueError: If the extension is unsupported.
        FileNotFoundError: If the input path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    extension = path.suffix.lower()
    if extension == ".csv":
        return pd.read_csv(path)
    if extension in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(
        f"Unsupported file extension '{extension}'. Use CSV (.csv) or Parquet (.parquet)."
    )


def load_sales_data(path: Path) -> pd.DataFrame:
    """Load sales data from disk.

    Args:
        path: Input sales file path.

    Returns:
        Sales dataframe.
    """
    file_path = Path(path)
    sales_df = _read_tabular_file(file_path)
    if DATE_COLUMN in sales_df.columns:
        sales_df[DATE_COLUMN] = pd.to_datetime(sales_df[DATE_COLUMN], errors="coerce")

    LOGGER.info("Loaded %d sales rows from %s", len(sales_df), file_path)
    return sales_df


def load_price_plan(path: Path) -> pd.DataFrame:
    """Load optional future price plan data from disk.

    Args:
        path: Input price plan path.

    Returns:
        Price plan dataframe.
    """
    file_path = Path(path)
    plan_df = _read_tabular_file(file_path)
    if DATE_COLUMN in plan_df.columns:
        plan_df[DATE_COLUMN] = pd.to_datetime(plan_df[DATE_COLUMN], errors="coerce")

    LOGGER.info("Loaded %d price-plan rows from %s", len(plan_df), file_path)
    return plan_df


def write_parquet(dataframe: pd.DataFrame, output_path: Path) -> Path:
    """Write a dataframe to parquet and return the output path.

    Args:
        dataframe: Data to write.
        output_path: Destination file path.

    Returns:
        The destination path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)
    LOGGER.info("Wrote %d rows to %s", len(dataframe), path)
    return path
