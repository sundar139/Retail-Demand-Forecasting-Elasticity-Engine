"""General utility helpers for the retail forecasting package."""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return the same path.

    Args:
        path: Directory path to create.

    Returns:
        The created/existing directory path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_dict_hash(payload: Mapping[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash for a dictionary-like payload.

    Args:
        payload: Dictionary-like payload.

    Returns:
        Hex digest string.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
