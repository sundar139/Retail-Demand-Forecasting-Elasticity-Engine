"""Data loading utilities for retail CSV ingestion."""

from pathlib import Path
import logging

import pandas as pd

from retail_forecasting.paths import build_project_paths

LOGGER = logging.getLogger(__name__)


def discover_raw_csv(
    input_path: Path | None = None,
    raw_dir: Path | None = None,
) -> Path:
    """Discover the raw CSV file for ingestion.

    Args:
        input_path: Optional explicit path to a CSV file.
        raw_dir: Optional directory used when input_path is not provided.

    Returns:
        Resolved CSV path.

    Raises:
        FileNotFoundError: If no valid CSV exists.
        ValueError: If multiple CSV files are found without explicit selection.
    """
    project_paths = build_project_paths()

    if input_path is not None:
        candidate = Path(input_path).expanduser()
        if not candidate.is_absolute():
            candidate = project_paths.project_root / candidate

        if not candidate.exists():
            raise FileNotFoundError(
                f"Input CSV not found: {candidate}. Place a CSV in data/raw or pass --input-path."
            )
        if candidate.suffix.lower() != ".csv":
            raise ValueError(f"Input file must be a CSV: {candidate}")
        return candidate

    search_dir = Path(raw_dir) if raw_dir is not None else project_paths.data_raw_dir
    csv_candidates = sorted(search_dir.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            "No CSV found in data/raw. Place the Kaggle retail CSV there or pass --input-path."
        )
    if len(csv_candidates) > 1:
        candidate_names = ", ".join(path.name for path in csv_candidates)
        raise ValueError(
            "Multiple CSV files found in data/raw; pass --input-path to choose one. "
            f"Candidates: {candidate_names}"
        )

    return csv_candidates[0]


def load_retail_csv(csv_path: Path) -> pd.DataFrame:
    """Load retail CSV data into a dataframe.

    Args:
        csv_path: Source CSV path.

    Returns:
        Loaded dataframe.
    """
    frame = pd.read_csv(csv_path, low_memory=False)
    LOGGER.info("Loaded %d rows and %d columns from %s", frame.shape[0], frame.shape[1], csv_path)
    return frame


def discover_and_load_csv(
    input_path: Path | None = None,
    raw_dir: Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Discover and load a raw CSV in one step.

    Args:
        input_path: Optional explicit CSV path.
        raw_dir: Optional data/raw directory override.

    Returns:
        Tuple of resolved path and loaded dataframe.
    """
    csv_path = discover_raw_csv(input_path=input_path, raw_dir=raw_dir)
    dataframe = load_retail_csv(csv_path)
    return csv_path, dataframe
