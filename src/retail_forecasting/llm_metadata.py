"""Shared helpers for LLM acceptance and metadata truth conditions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from retail_forecasting.paths import ProjectPaths, build_project_paths
from retail_forecasting.preprocessing import load_json


@dataclass(frozen=True, slots=True)
class LLMUsageFacts:
    """Normalized LLM metadata used across training, evaluation, and reporting."""

    llm_requested: bool
    ollama_reachable: bool | None
    planner_model_available: bool | None
    llm_feature_file_exists: bool
    llm_feature_file_path: str
    llm_output_feature_count: int
    llm_features_actually_used: bool
    llm_feature_columns_used: list[str]


def derive_llm_usage_facts(
    llm_requested: bool,
    llm_feature_columns_used: Sequence[str] | str | None,
    project_paths: ProjectPaths | None = None,
) -> LLMUsageFacts:
    """Build a single source-of-truth LLM metadata snapshot.

    LLM features are considered actually used only when:
    1) a features_llm artifact exists,
    2) summary reports output feature columns,
    3) at least one of those columns is used downstream in a model matrix.
    """
    paths = project_paths if project_paths is not None else build_project_paths()
    summary_payload = _load_llm_summary(paths)

    columns_used = _normalize_columns(llm_feature_columns_used)

    if not llm_requested:
        feature_path = _resolve_llm_feature_path(paths, summary_payload)
        return LLMUsageFacts(
            llm_requested=False,
            ollama_reachable=None,
            planner_model_available=None,
            llm_feature_file_exists=feature_path.exists(),
            llm_feature_file_path=str(feature_path),
            llm_output_feature_count=0,
            llm_features_actually_used=False,
            llm_feature_columns_used=columns_used,
        )

    ollama_reachable = _coerce_optional_bool(
        summary_payload.get("ollama_reachable") if summary_payload is not None else None
    )
    planner_model_available = _coerce_optional_bool(
        summary_payload.get("planner_model_available") if summary_payload is not None else None
    )
    output_feature_count = _coerce_int(
        summary_payload.get("output_feature_count") if summary_payload is not None else None
    )

    feature_path = _resolve_llm_feature_path(paths, summary_payload)
    feature_exists = feature_path.exists()

    llm_features_actually_used = bool(
        feature_exists and output_feature_count > 0 and columns_used
    )

    return LLMUsageFacts(
        llm_requested=True,
        ollama_reachable=ollama_reachable,
        planner_model_available=planner_model_available,
        llm_feature_file_exists=feature_exists,
        llm_feature_file_path=str(feature_path),
        llm_output_feature_count=output_feature_count,
        llm_features_actually_used=llm_features_actually_used,
        llm_feature_columns_used=columns_used,
    )


def llm_usage_facts_to_dict(facts: LLMUsageFacts) -> dict[str, object]:
    """Convert usage facts to a JSON-friendly dictionary."""
    return {
        "llm_requested": facts.llm_requested,
        "ollama_reachable": facts.ollama_reachable,
        "planner_model_available": facts.planner_model_available,
        "llm_feature_file_exists": facts.llm_feature_file_exists,
        "llm_feature_file_path": facts.llm_feature_file_path,
        "llm_output_feature_count": facts.llm_output_feature_count,
        "llm_features_actually_used": facts.llm_features_actually_used,
        "llm_feature_columns_used": list(facts.llm_feature_columns_used),
    }


def _load_llm_summary(project_paths: ProjectPaths) -> dict[str, object] | None:
    summary_path = project_paths.artifacts_dir / "llm_features_summary.json"
    if not summary_path.exists():
        return None

    payload = load_json(summary_path)
    return payload if isinstance(payload, dict) else None


def _resolve_llm_feature_path(
    project_paths: ProjectPaths,
    summary_payload: dict[str, object] | None,
) -> Path:
    default_path = project_paths.data_processed_dir / "features_llm.parquet"

    if summary_payload is None:
        return default_path

    output_paths = summary_payload.get("output_paths")
    if not isinstance(output_paths, dict):
        return default_path

    candidate_value = output_paths.get("features_llm_parquet")
    if not isinstance(candidate_value, str) or not candidate_value.strip():
        return default_path

    candidate_path = Path(candidate_value)
    if not candidate_path.is_absolute():
        candidate_path = project_paths.project_root / candidate_path
    return candidate_path


def _normalize_columns(values: Sequence[str] | str | None) -> list[str]:
    if values is None:
        return []

    if isinstance(values, str):
        parts = [part.strip() for part in values.split("|")]
        return [part for part in parts if part]

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized


def _coerce_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def _coerce_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default
