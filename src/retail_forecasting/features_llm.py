"""Safe LLM-assisted feature planning and deterministic materialization."""

from collections.abc import Sequence
from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd

from retail_forecasting.feature_spec import (
    BinaryFlagFeatureSpec,
    CalendarFeatureSpec,
    DifferenceFeatureSpec,
    FeatureSpec,
    InteractionFeatureSpec,
    LagFeatureSpec,
    RatioFeatureSpec,
    RollingFeatureSpec,
    extract_feature_name_from_raw,
    extract_raw_specs_from_payload,
    feature_plan_json_schema,
    validate_feature_plan_specs,
)
from retail_forecasting.features_common import ensure_group_date_sort_order, safe_ratio
from retail_forecasting.features_manual import expected_manual_feature_names
from retail_forecasting.ollama_client import OllamaClient
from retail_forecasting.paths import ProjectPaths, build_project_paths
from retail_forecasting.preprocessing import load_json, write_json, write_parquet
from retail_forecasting.schemas import DATE_COLUMN, GROUP_COLUMNS, OPTIONAL_CANONICAL_COLUMNS, REQUIRED_CANONICAL_COLUMNS

LOGGER = logging.getLogger(__name__)

_RAW_PLAN_FILENAME = "llm_feature_plan_raw.json"
_VALIDATED_PLAN_FILENAME = "llm_feature_plan_validated.json"
_SUMMARY_FILENAME = "llm_features_summary.json"
_FEATURES_OUTPUT_FILENAME = "features_llm.parquet"

_LEAKAGE_SAFETY_STATEMENT = (
    "LLM plans are translated into deterministic transformations only. Lag and rolling features "
    "always rely on prior rows using positive shifts, and no transformation may read future values."
)

_BASE_OUTPUT_COLUMNS: tuple[str, ...] = (
    DATE_COLUMN,
    GROUP_COLUMNS[0],
    GROUP_COLUMNS[1],
    "units_sold",
    "price",
)

_PROMPT_SYSTEM_PATH = Path("prompts") / "feature_planner_system.txt"
_PROMPT_USER_TEMPLATE_PATH = Path("prompts") / "feature_planner_user_template.txt"


def plan_llm_features_pipeline(
    input_path: Path | None = None,
    planner: OllamaClient | None = None,
    include_manual_input: bool = False,
) -> dict[str, Path]:
    """Generate and validate LLM feature plan artifacts without materialization."""
    return _run_llm_feature_pipeline(
        input_path=input_path,
        planner=planner,
        include_manual_input=include_manual_input,
        materialize_features=False,
    )


def build_llm_features_pipeline(
    input_path: Path | None = None,
    planner: OllamaClient | None = None,
    include_manual_input: bool = False,
    output_path: Path | None = None,
) -> dict[str, Path]:
    """Generate, validate, and materialize deterministic LLM feature artifacts."""
    return _run_llm_feature_pipeline(
        input_path=input_path,
        planner=planner,
        include_manual_input=include_manual_input,
        materialize_features=True,
        output_path=output_path,
    )


def load_llm_features_summary(summary_path: Path | None = None) -> dict[str, object]:
    """Load the latest LLM feature summary artifact."""
    paths = build_project_paths()
    target = paths.artifacts_dir / _SUMMARY_FILENAME if summary_path is None else Path(summary_path)
    if not target.is_absolute():
        target = paths.project_root / target

    if not target.exists():
        raise FileNotFoundError(
            f"Missing LLM feature summary: {target}. Run 'plan-llm-features' or 'build-llm-features' first."
        )

    return load_json(target)


def _run_llm_feature_pipeline(
    input_path: Path | None,
    planner: OllamaClient | None,
    include_manual_input: bool,
    materialize_features: bool,
    output_path: Path | None = None,
) -> dict[str, Path]:
    paths = build_project_paths()
    source_path, source_df = load_llm_source_data(
        input_path=input_path,
        include_manual_input=include_manual_input,
        project_paths=paths,
    )

    manual_namespace = load_manual_feature_namespace(paths)
    available_columns = [str(column_name) for column_name in source_df.columns]

    system_prompt, user_prompt = render_feature_planner_prompts(
        available_columns=available_columns,
        manual_feature_names=sorted(manual_namespace),
        project_paths=paths,
    )

    client = planner if planner is not None else OllamaClient()
    planner_response = client.plan_feature_specs(system_prompt=system_prompt, user_prompt=user_prompt)

    raw_plan_payload = planner_response.parsed_json if planner_response.parsed_json is not None else {"specs": []}
    raw_specs = extract_raw_specs_from_payload(raw_plan_payload)

    proposed_names = {
        extract_feature_name_from_raw(raw_spec, fallback_index=index)
        for index, raw_spec in enumerate(raw_specs, start=1)
    }
    proposed_overlap_with_manual = sorted(proposed_names & manual_namespace)

    accepted_specs, rejected_specs = validate_feature_plan_specs(
        raw_specs=raw_specs,
        available_columns=available_columns,
        existing_feature_names=available_columns,
        blocked_feature_names=sorted(manual_namespace),
    )

    if planner_response.error:
        rejected_specs.insert(
            0,
            {
                "index": 0,
                "feature_name": "planner_request",
                "operation": "planner_error",
                "reason": planner_response.error,
            },
        )

    validated_plan_payload = {
        "plan_version": "1.0",
        "planner_host": planner_response.host,
        "planner_model": planner_response.model,
        "ollama_reachable": planner_response.reachable,
        "planner_model_available": planner_response.planner_model_available,
        "source_dataset_path": str(source_path),
        "available_columns": available_columns,
        "raw_spec_count": len(raw_specs),
        "accepted_specs": [spec.model_dump(mode="json") for spec in accepted_specs],
        "rejected_specs": rejected_specs,
        "proposed_overlap_with_manual_features": proposed_overlap_with_manual,
    }

    raw_artifact_payload = {
        "planner_host": planner_response.host,
        "planner_model": planner_response.model,
        "ollama_reachable": planner_response.reachable,
        "planner_model_available": planner_response.planner_model_available,
        "planner_error": planner_response.error,
        "source_dataset_path": str(source_path),
        "prompt_system_path": str(paths.project_root / _PROMPT_SYSTEM_PATH),
        "prompt_user_template_path": str(paths.project_root / _PROMPT_USER_TEMPLATE_PATH),
        "feature_plan_schema": feature_plan_json_schema(),
        "raw_response_text": planner_response.raw_response_text,
        "parsed_json": raw_plan_payload,
    }

    raw_plan_path = write_json(raw_artifact_payload, paths.artifacts_dir / _RAW_PLAN_FILENAME)
    validated_plan_path = write_json(validated_plan_payload, paths.artifacts_dir / _VALIDATED_PLAN_FILENAME)

    output_paths: dict[str, Path] = {
        "llm_feature_plan_raw_json": raw_plan_path,
        "llm_feature_plan_validated_json": validated_plan_path,
    }

    materialized_features: list[str] = []
    materialization_rejections: list[dict[str, object]] = []
    resolved_output = _resolve_features_output_path(paths, output_path)
    if materialize_features:
        feature_frame, materialized_features, materialization_rejections = materialize_llm_features(
            source_df=source_df,
            specs=accepted_specs,
        )
        features_path = write_parquet(feature_frame, resolved_output)
        output_paths["features_llm_parquet"] = features_path

    all_rejections = rejected_specs + materialization_rejections
    summary_payload = {
        "llm_requested": True,
        "planner_model_used": planner_response.model,
        "planner_host_used": planner_response.host,
        "planner_model_available": planner_response.planner_model_available,
        "source_dataset_path": str(source_path),
        "raw_spec_count": len(raw_specs),
        "accepted_spec_count": len(accepted_specs),
        "rejected_spec_count": len(all_rejections),
        "rejected_specs": all_rejections,
        "output_feature_count": len(materialized_features),
        "output_feature_names": materialized_features,
        "llm_feature_file_path": str(resolved_output),
        "llm_feature_file_exists": resolved_output.exists(),
        "overlap_with_manual_features": proposed_overlap_with_manual,
        "leakage_safety_statement": _LEAKAGE_SAFETY_STATEMENT,
        "ollama_reachable": planner_response.reachable,
        "planner_error": planner_response.error,
        "materialization_enabled": materialize_features,
        "output_paths": {name: str(path_value) for name, path_value in output_paths.items()},
    }

    summary_path = write_json(summary_payload, paths.artifacts_dir / _SUMMARY_FILENAME)
    output_paths["llm_features_summary_json"] = summary_path

    LOGGER.info(
        "LLM feature phase complete: raw_specs=%d accepted=%d output_features=%d",
        len(raw_specs),
        len(accepted_specs),
        len(materialized_features),
    )
    return output_paths


def load_llm_source_data(
    input_path: Path | None = None,
    include_manual_input: bool = False,
    project_paths: ProjectPaths | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Load source parquet for Phase 4 planning and materialization."""
    paths = project_paths if project_paths is not None else build_project_paths()

    if input_path is not None:
        candidate = Path(input_path).expanduser()
        if not candidate.is_absolute():
            candidate = paths.project_root / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"LLM feature input parquet not found: {candidate}")
        source_path = candidate
    else:
        manual_candidate = paths.data_processed_dir / "features_manual.parquet"
        cleaned_candidate = paths.data_interim_dir / "cleaned_retail.parquet"

        if include_manual_input and manual_candidate.exists():
            source_path = manual_candidate
        else:
            source_path = cleaned_candidate

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source data for LLM features was not found at {source_path}. "
            "Run 'prepare-data' first, or pass --input-path."
        )

    frame = pd.read_parquet(source_path)
    LOGGER.info(
        "Loaded LLM source data from %s with %d rows and %d columns",
        source_path,
        frame.shape[0],
        frame.shape[1],
    )
    return source_path, frame


def render_feature_planner_prompts(
    available_columns: Sequence[str],
    manual_feature_names: Sequence[str],
    project_paths: ProjectPaths | None = None,
) -> tuple[str, str]:
    """Render system and user prompts for schema-aware JSON feature planning."""
    paths = project_paths if project_paths is not None else build_project_paths()

    system_path = paths.project_root / _PROMPT_SYSTEM_PATH
    user_template_path = paths.project_root / _PROMPT_USER_TEMPLATE_PATH

    system_prompt = system_path.read_text(encoding="utf-8")
    user_template = user_template_path.read_text(encoding="utf-8")

    present_optional = sorted(set(available_columns) & set(OPTIONAL_CANONICAL_COLUMNS))
    absent_optional = sorted(set(OPTIONAL_CANONICAL_COLUMNS) - set(available_columns))

    user_prompt = user_template.format(
        required_columns_json=json.dumps(sorted(REQUIRED_CANONICAL_COLUMNS)),
        optional_present_json=json.dumps(present_optional),
        optional_absent_json=json.dumps(absent_optional),
        available_columns_json=json.dumps(sorted(set(available_columns))),
        manual_feature_names_json=json.dumps(sorted(set(manual_feature_names))),
        feature_plan_schema_json=json.dumps(feature_plan_json_schema(), indent=2),
    )

    return system_prompt, user_prompt


def materialize_llm_features(
    source_df: pd.DataFrame,
    specs: Sequence[FeatureSpec],
) -> tuple[pd.DataFrame, list[str], list[dict[str, object]]]:
    """Materialize validated specs into deterministic pandas features."""
    frame = ensure_group_date_sort_order(source_df)

    created_features: list[str] = []
    rejected: list[dict[str, object]] = []

    for spec in specs:
        if spec.feature_name in frame.columns:
            rejected.append(
                {
                    "feature_name": spec.feature_name,
                    "operation": spec.operation,
                    "reason": "feature name collision in materialization frame",
                }
            )
            continue

        try:
            series = _materialize_feature_series(frame, spec)
        except ValueError as exc:
            rejected.append(
                {
                    "feature_name": spec.feature_name,
                    "operation": spec.operation,
                    "reason": str(exc),
                }
            )
            continue

        frame[spec.feature_name] = series
        created_features.append(spec.feature_name)

    output_columns = _ordered_unique(list(_BASE_OUTPUT_COLUMNS) + created_features)
    for required_column in _BASE_OUTPUT_COLUMNS:
        if required_column not in frame.columns:
            raise ValueError(
                f"LLM materialization requires base column '{required_column}' to be present"
            )
    output_frame = frame[output_columns].copy(deep=True)
    return output_frame, created_features, rejected


def load_manual_feature_namespace(project_paths: ProjectPaths | None = None) -> set[str]:
    """Load manual feature namespace used for overlap and collision checks."""
    paths = project_paths if project_paths is not None else build_project_paths()

    namespace = set(expected_manual_feature_names())
    manual_path = paths.data_processed_dir / "features_manual.parquet"
    if manual_path.exists():
        try:
            columns = pd.read_parquet(manual_path).columns.tolist()
            for column_name in columns:
                if column_name not in _BASE_OUTPUT_COLUMNS:
                    namespace.add(column_name)
        except Exception:  # noqa: BLE001
            LOGGER.warning("Unable to inspect manual feature artifact columns at %s", manual_path)

    return namespace


def _materialize_feature_series(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    if isinstance(spec, LagFeatureSpec):
        source = _numeric_series(frame, spec.source_column)
        grouped = source.groupby([frame[column_name] for column_name in spec.group_by], sort=False)
        return grouped.shift(spec.lag)

    if isinstance(spec, RollingFeatureSpec):
        source = _numeric_series(frame, spec.source_column)
        grouped = source.groupby([frame[column_name] for column_name in spec.group_by], sort=False)
        return grouped.transform(
            lambda series, spec_window=spec.window, spec_shift=spec.shift, spec_agg=spec.aggregation: _shifted_rolling(
                series=series,
                window=spec_window,
                shift=spec_shift,
                aggregation=spec_agg,
            )
        )

    if isinstance(spec, RatioFeatureSpec):
        numerator = _numeric_series(frame, spec.numerator_column)
        denominator = _numeric_series(frame, spec.denominator_column)
        return safe_ratio(numerator, denominator)

    if isinstance(spec, DifferenceFeatureSpec):
        minuend = _numeric_series(frame, spec.minuend_column)
        subtrahend = _numeric_series(frame, spec.subtrahend_column)
        return (minuend - subtrahend).astype("float64")

    if isinstance(spec, InteractionFeatureSpec):
        left = _numeric_series(frame, spec.left_column)
        right = _numeric_series(frame, spec.right_column)
        return (left * right).astype("float64")

    if isinstance(spec, BinaryFlagFeatureSpec):
        source = _numeric_series(frame, spec.source_column)
        comparator_map = {
            "gt": source > spec.threshold,
            "ge": source >= spec.threshold,
            "lt": source < spec.threshold,
            "le": source <= spec.threshold,
            "eq": source == spec.threshold,
            "ne": source != spec.threshold,
        }
        condition = comparator_map[spec.comparator]
        result = pd.Series(spec.false_value, index=frame.index, dtype="float64")
        result.loc[condition] = float(spec.true_value)
        result.loc[source.isna()] = np.nan
        return result

    if isinstance(spec, CalendarFeatureSpec):
        calendar_component = _calendar_component(frame)
        if spec.interact_with_column is None:
            return calendar_component[spec.calendar_component]

        source = _numeric_series(frame, spec.interact_with_column)
        return (calendar_component[spec.calendar_component] * source).astype("float64")

    raise ValueError(f"Unsupported operation type during materialization: {spec.operation}")


def _calendar_component(frame: pd.DataFrame) -> dict[str, pd.Series]:
    date_series = pd.to_datetime(frame[DATE_COLUMN], errors="coerce").dt.floor("D")
    if date_series.isna().any():
        raise ValueError("Date column contains invalid values for calendar feature materialization")

    return {
        "day_of_week": date_series.dt.dayofweek.astype("float64"),
        "day_of_month": date_series.dt.day.astype("float64"),
        "day_of_year": date_series.dt.dayofyear.astype("float64"),
        "week_of_year": date_series.dt.isocalendar().week.astype("float64"),
        "month": date_series.dt.month.astype("float64"),
        "quarter": date_series.dt.quarter.astype("float64"),
        "is_weekend": (date_series.dt.dayofweek >= 5).astype("float64"),
        "is_month_start": date_series.dt.is_month_start.astype("float64"),
        "is_month_end": date_series.dt.is_month_end.astype("float64"),
    }


def _numeric_series(frame: pd.DataFrame, column_name: str) -> pd.Series:
    converted = pd.to_numeric(frame[column_name], errors="coerce").astype("float64")
    if converted.isna().all():
        raise ValueError(f"Column '{column_name}' has no numeric values for deterministic materialization")
    return converted


def _shifted_rolling(series: pd.Series, window: int, shift: int, aggregation: str) -> pd.Series:
    shifted = series.shift(shift)
    rolling = shifted.rolling(window=window, min_periods=window)

    if aggregation == "mean":
        return rolling.mean()
    if aggregation == "std":
        return rolling.std()
    if aggregation == "min":
        return rolling.min()
    if aggregation == "max":
        return rolling.max()

    raise ValueError(f"Unsupported rolling aggregation '{aggregation}'")


def _resolve_features_output_path(project_paths: ProjectPaths, output_path: Path | None) -> Path:
    candidate = (
        project_paths.data_processed_dir / _FEATURES_OUTPUT_FILENAME
        if output_path is None
        else Path(output_path).expanduser()
    )
    if not candidate.is_absolute():
        candidate = project_paths.project_root / candidate
    return candidate


def _ordered_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered
