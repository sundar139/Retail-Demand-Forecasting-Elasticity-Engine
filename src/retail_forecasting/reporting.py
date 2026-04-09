"""Phase 7 reporting, manifest, and consistency reconciliation utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

from retail_forecasting.llm_metadata import LLMUsageFacts, derive_llm_usage_facts, llm_usage_facts_to_dict
from retail_forecasting.paths import ProjectPaths, build_project_paths
from retail_forecasting.preprocessing import load_json, write_json

DEFAULT_OPTIMIZE_METRIC = "wmape"


@dataclass(frozen=True, slots=True)
class ReportingConfig:
    """Configuration for report and manifest generation workflows."""

    optimize_metric: str = DEFAULT_OPTIMIZE_METRIC
    force_refresh_report: bool = False
    report_output_path: Path | None = None
    run_config_values: dict[str, object] | None = None
    stage_records: list[dict[str, object]] | None = None
    upstream_warnings: list[str] | None = None


def generate_reporting_artifacts(config: ReportingConfig | None = None) -> dict[str, Path]:
    """Generate summary artifacts, manifest, and final markdown report from source artifacts."""
    run_config = config if config is not None else ReportingConfig()
    optimize_metric = _normalize_metric_name(run_config.optimize_metric)

    paths = build_project_paths()
    warnings: list[str] = list(run_config.upstream_warnings or [])
    outputs: dict[str, Path] = {}
    llm_summary = _load_json_if_exists(paths.artifacts_dir / "llm_features_summary.json")

    validation_metrics = _read_csv_if_exists(paths.artifacts_dir / "forecast_metrics_validation.csv")
    test_metrics = _read_csv_if_exists(paths.artifacts_dir / "forecast_metrics_test.csv")
    segment_metrics = _read_csv_if_exists(paths.artifacts_dir / "forecast_segment_metrics.csv")

    forecast_summary = _build_forecast_metrics_summary(
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        optimize_metric=optimize_metric,
        warnings=warnings,
    )
    forecast_summary_path = paths.artifacts_dir / "forecast_metrics_summary.csv"
    forecast_summary.to_csv(forecast_summary_path, index=False)
    outputs["forecast_metrics_summary_csv"] = forecast_summary_path

    training_registry = _read_csv_if_exists(paths.artifacts_dir / "model_training_registry.csv")
    best_registry = _build_consistent_best_model_registry(
        forecast_summary=forecast_summary,
        segment_metrics=segment_metrics,
        training_registry=training_registry,
        project_paths=paths,
        optimize_metric=optimize_metric,
        existing_registry_path=paths.artifacts_dir / "best_model_registry.csv",
        warnings=warnings,
    )

    best_registry_path = paths.artifacts_dir / "best_model_registry.csv"
    best_registry.to_csv(best_registry_path, index=False)
    outputs["best_model_registry_csv"] = best_registry_path

    elasticity_summary = _build_elasticity_summary(paths=paths, warnings=warnings)
    elasticity_summary_path = paths.artifacts_dir / "elasticity_summary.csv"
    elasticity_summary.to_csv(elasticity_summary_path, index=False)
    outputs["elasticity_summary_csv"] = elasticity_summary_path

    best_overall = _select_best_overall_row(forecast_summary=forecast_summary, best_registry=best_registry)
    llm_requested = _resolve_llm_requested(run_config.run_config_values, llm_summary)
    llm_columns_for_best = _extract_best_llm_feature_columns(best_registry)
    llm_usage = derive_llm_usage_facts(
        llm_requested=llm_requested,
        llm_feature_columns_used=llm_columns_for_best,
        project_paths=paths,
    )
    elasticity_overview = _build_elasticity_overview(
        elasticity_summary=elasticity_summary,
        project_paths=paths,
    )

    final_summary_payload = _build_final_project_summary_payload(
        paths=paths,
        forecast_summary=forecast_summary,
        best_overall=best_overall,
        elasticity_summary=elasticity_summary,
        llm_summary=llm_summary,
        llm_usage=llm_usage,
        elasticity_overview=elasticity_overview,
        warnings=warnings,
    )
    final_summary_path = write_json(final_summary_payload, paths.artifacts_dir / "final_project_summary.json")
    outputs["final_project_summary_json"] = final_summary_path

    manifest_payload = _build_run_manifest_payload(
        paths=paths,
        optimize_metric=optimize_metric,
        best_overall=best_overall,
        warnings=warnings,
        output_paths=outputs,
        llm_summary=llm_summary,
        llm_usage=llm_usage,
        elasticity_overview=elasticity_overview,
        run_config_values=run_config.run_config_values,
        stage_records=run_config.stage_records,
    )
    manifest_path = write_json(manifest_payload, paths.artifacts_dir / "run_manifest.json")
    outputs["run_manifest_json"] = manifest_path

    acceptance_payload = _build_acceptance_summary_payload(
        optimize_metric=optimize_metric,
        best_overall=best_overall,
        llm_usage=llm_usage,
        elasticity_overview=elasticity_overview,
        output_paths=outputs,
    )
    acceptance_path = write_json(acceptance_payload, paths.artifacts_dir / "acceptance_summary.json")
    outputs["acceptance_summary_json"] = acceptance_path

    report_path = _resolve_report_output_path(paths, run_config.report_output_path)
    report_content = _render_final_markdown_report(
        paths=paths,
        forecast_summary=forecast_summary,
        best_overall=best_overall,
        elasticity_summary=elasticity_summary,
        llm_summary=llm_summary,
        llm_usage=llm_usage,
        elasticity_overview=elasticity_overview,
        warnings=warnings,
        outputs=outputs,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")
    outputs["final_run_report_markdown"] = report_path

    return outputs


def _normalize_metric_name(metric_name: str) -> str:
    normalized = metric_name.strip().lower()
    allowed = {"wmape", "mape", "mae", "rmse"}
    if normalized not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported optimize metric '{metric_name}'. Allowed values: {allowed_text}")
    return normalized


def _build_forecast_metrics_summary(
    validation_metrics: pd.DataFrame | None,
    test_metrics: pd.DataFrame | None,
    optimize_metric: str,
    warnings: list[str],
) -> pd.DataFrame:
    summary_columns = [
        "model_name",
        "model_family",
        "validation_mape",
        "validation_wmape",
        "validation_mae",
        "validation_rmse",
        "test_mape",
        "test_wmape",
        "test_mae",
        "test_rmse",
        "validation_rank",
        "is_best_overall",
    ]

    if validation_metrics is None or validation_metrics.empty:
        warnings.append(
            "Forecast validation metrics are missing; forecast summary will be empty until evaluation artifacts exist"
        )
        return pd.DataFrame(columns=summary_columns)

    validation = validation_metrics.copy(deep=True)
    expected_validation = {"model_name", "model_family", "mape", "wmape", "mae", "rmse"}
    missing_validation = sorted(expected_validation - set(validation.columns))
    if missing_validation:
        raise ValueError(
            "forecast_metrics_validation.csv is missing columns: " + ", ".join(missing_validation)
        )

    validation = validation.rename(
        columns={
            "mape": "validation_mape",
            "wmape": "validation_wmape",
            "mae": "validation_mae",
            "rmse": "validation_rmse",
        }
    )

    test_table: pd.DataFrame
    if test_metrics is None or test_metrics.empty:
        warnings.append("Forecast test metrics are missing; summary will include NaN test metrics")
        test_table = pd.DataFrame(columns=["model_name", "model_family", "test_mape", "test_wmape", "test_mae", "test_rmse"])
    else:
        expected_test = {"model_name", "model_family", "mape", "wmape", "mae", "rmse"}
        missing_test = sorted(expected_test - set(test_metrics.columns))
        if missing_test:
            raise ValueError("forecast_metrics_test.csv is missing columns: " + ", ".join(missing_test))
        test_table = test_metrics.copy(deep=True).rename(
            columns={
                "mape": "test_mape",
                "wmape": "test_wmape",
                "mae": "test_mae",
                "rmse": "test_rmse",
            }
        )

    merged = validation[["model_name", "model_family", "validation_mape", "validation_wmape", "validation_mae", "validation_rmse"]].merge(
        test_table[["model_name", "model_family", "test_mape", "test_wmape", "test_mae", "test_rmse"]],
        on=["model_name", "model_family"],
        how="left",
    )

    metric_column = f"validation_{optimize_metric}"
    metric_values = pd.to_numeric(merged[metric_column], errors="coerce").astype("float64")
    merged = merged.loc[metric_values.notna()].copy()

    if merged.empty:
        warnings.append("No finite validation metric values found after summary merge")
        return pd.DataFrame(columns=summary_columns)

    merged = merged.sort_values(by=[metric_column, "model_name"], kind="mergesort").reset_index(drop=True)
    merged["validation_rank"] = np.arange(1, len(merged) + 1, dtype="int64")
    merged["is_best_overall"] = merged["validation_rank"] == 1

    for column_name in [
        "validation_mape",
        "validation_wmape",
        "validation_mae",
        "validation_rmse",
        "test_mape",
        "test_wmape",
        "test_mae",
        "test_rmse",
    ]:
        merged[column_name] = pd.to_numeric(merged[column_name], errors="coerce").astype("float64")

    return merged[summary_columns]


def _build_consistent_best_model_registry(
    forecast_summary: pd.DataFrame,
    segment_metrics: pd.DataFrame | None,
    training_registry: pd.DataFrame | None,
    project_paths: ProjectPaths,
    optimize_metric: str,
    existing_registry_path: Path,
    warnings: list[str],
) -> pd.DataFrame:
    columns = [
        "selected_at_utc",
        "scope",
        "segment_key",
        "optimize_metric",
        "model_name",
        "model_family",
        "validation_mape",
        "validation_wmape",
        "validation_mae",
        "validation_rmse",
        "test_mape",
        "test_wmape",
        "test_mae",
        "test_rmse",
        "training_source",
        "feature_count",
        "llm_features_used",
        "llm_feature_columns",
        "model_artifact_path",
    ]

    if forecast_summary.empty:
        existing = _read_csv_if_exists(existing_registry_path)
        if existing is not None:
            return existing
        return pd.DataFrame(columns=columns)

    selected_at = _utc_now_iso()
    best = forecast_summary.iloc[0]

    metadata_by_model = _build_training_metadata_map(training_registry, project_paths=project_paths)
    best_metadata = metadata_by_model.get(str(best["model_name"]), _baseline_metadata())

    rows: list[dict[str, object]] = [
        {
            "selected_at_utc": selected_at,
            "scope": "overall",
            "segment_key": "all",
            "optimize_metric": optimize_metric,
            "model_name": str(best["model_name"]),
            "model_family": str(best["model_family"]),
            "validation_mape": float(best["validation_mape"]),
            "validation_wmape": float(best["validation_wmape"]),
            "validation_mae": float(best["validation_mae"]),
            "validation_rmse": float(best["validation_rmse"]),
            "test_mape": _to_float(best.get("test_mape")),
            "test_wmape": _to_float(best.get("test_wmape")),
            "test_mae": _to_float(best.get("test_mae")),
            "test_rmse": _to_float(best.get("test_rmse")),
            "training_source": str(best_metadata.get("training_source", "unknown")),
            "feature_count": _coerce_int(best_metadata.get("feature_count", 0)),
            "llm_features_used": bool(best_metadata.get("llm_features_used", False)),
            "llm_feature_columns": _normalize_llm_columns(best_metadata.get("llm_feature_columns", "")),
            "model_artifact_path": str(best_metadata.get("model_artifact_path", "")),
        }
    ]

    if segment_metrics is not None and not segment_metrics.empty:
        required = {"split", "segment_key", "model_name", "model_family", "mape", "wmape", "mae", "rmse"}
        missing = sorted(required - set(segment_metrics.columns))
        if missing:
            warnings.append(
                "forecast_segment_metrics.csv is missing required columns for per-segment winners: "
                + ", ".join(missing)
            )
        else:
            validation_segment = segment_metrics.loc[
                segment_metrics["split"].astype("string") == "validation"
            ].copy()
            metric_column = optimize_metric
            validation_segment[metric_column] = pd.to_numeric(
                validation_segment[metric_column], errors="coerce"
            )
            validation_segment = validation_segment.loc[validation_segment[metric_column].notna()].copy()

            for segment_key, segment_frame in validation_segment.groupby("segment_key", sort=True):
                segment_best = segment_frame.sort_values(by=[metric_column], kind="mergesort").iloc[0]
                model_name = str(segment_best["model_name"])
                model_metadata = metadata_by_model.get(model_name, _baseline_metadata())
                llm_columns = _normalize_llm_columns(model_metadata.get("llm_feature_columns", ""))
                rows.append(
                    {
                        "selected_at_utc": selected_at,
                        "scope": "per-product",
                        "segment_key": str(segment_key),
                        "optimize_metric": optimize_metric,
                        "model_name": model_name,
                        "model_family": str(segment_best["model_family"]),
                        "validation_mape": float(segment_best["mape"]),
                        "validation_wmape": float(segment_best["wmape"]),
                        "validation_mae": float(segment_best["mae"]),
                        "validation_rmse": float(segment_best["rmse"]),
                        "test_mape": float("nan"),
                        "test_wmape": float("nan"),
                        "test_mae": float("nan"),
                        "test_rmse": float("nan"),
                        "training_source": str(model_metadata.get("training_source", "unknown")),
                        "feature_count": _coerce_int(model_metadata.get("feature_count", 0)),
                        "llm_features_used": bool(model_metadata.get("llm_features_used", False)),
                        "llm_feature_columns": llm_columns,
                        "model_artifact_path": str(model_metadata.get("model_artifact_path", "")),
                    }
                )

    reconciled = pd.DataFrame.from_records(rows, columns=columns)

    existing_registry = _read_csv_if_exists(existing_registry_path)
    if existing_registry is not None and not existing_registry.empty:
        existing_overall = existing_registry.loc[existing_registry["scope"] == "overall"].head(1)
        if not existing_overall.empty:
            existing_model = str(existing_overall.iloc[0].get("model_name", ""))
            reconciled_model = str(reconciled.iloc[0].get("model_name", ""))
            if existing_model != reconciled_model:
                warnings.append(
                    "Existing best_model_registry overall winner differed from validation metrics source-of-truth and was reconciled"
                )

    return reconciled


def _build_training_metadata_map(
    training_registry: pd.DataFrame | None,
    project_paths: ProjectPaths,
) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {
        "naive_last": _baseline_metadata(),
        "seasonal_naive_7": _baseline_metadata(),
        "rolling_mean_7": _baseline_metadata(),
        "rolling_mean_28": _baseline_metadata(),
        "median_expanding": _baseline_metadata(),
    }

    if training_registry is None or training_registry.empty:
        return metadata

    required = {
        "model_name",
        "status",
        "model_path",
        "training_source",
        "feature_count",
        "llm_feature_columns",
        "trained_at_utc",
    }
    missing = sorted(required - set(training_registry.columns))
    if missing:
        return metadata

    successful = training_registry.loc[
        training_registry["status"].astype("string") == "success"
    ].copy()
    if successful.empty:
        return metadata

    successful = successful.sort_values(by=["trained_at_utc"], kind="mergesort")
    successful = successful.drop_duplicates(subset=["model_name"], keep="last")

    for row in successful.itertuples(index=False):
        llm_columns = _normalize_llm_columns(getattr(row, "llm_feature_columns", ""))
        llm_requested_value = getattr(row, "llm_requested", None)
        if llm_requested_value is None:
            llm_requested_value = getattr(row, "llm_flag_requested", None)

        llm_requested = _coerce_bool(llm_requested_value)
        if llm_requested is None:
            llm_requested = bool(llm_columns.strip())

        llm_usage = derive_llm_usage_facts(
            llm_requested=llm_requested,
            llm_feature_columns_used=llm_columns,
            project_paths=project_paths,
        )

        normalized_columns = (
            "|".join(llm_usage.llm_feature_columns_used)
            if llm_usage.llm_features_actually_used
            else ""
        )
        metadata[str(getattr(row, "model_name"))] = {
            "training_source": str(getattr(row, "training_source", "unknown")),
            "feature_count": _coerce_int(getattr(row, "feature_count", 0)),
            "llm_requested": bool(llm_usage.llm_requested),
            "llm_feature_columns": normalized_columns,
            "llm_features_used": bool(llm_usage.llm_features_actually_used),
            "llm_output_feature_count": int(llm_usage.llm_output_feature_count),
            "ollama_reachable": llm_usage.ollama_reachable,
            "planner_model_available": llm_usage.planner_model_available,
            "model_artifact_path": str(getattr(row, "model_path", "")),
        }

    return metadata


def _baseline_metadata() -> dict[str, object]:
    return {
        "training_source": "split_feature_artifacts",
        "feature_count": 0,
        "llm_requested": False,
        "llm_features_used": False,
        "llm_feature_columns": "",
        "llm_output_feature_count": 0,
        "ollama_reachable": None,
        "planner_model_available": None,
        "model_artifact_path": "",
    }


def _build_elasticity_summary(paths: ProjectPaths, warnings: list[str]) -> pd.DataFrame:
    columns = [
        "segment_key",
        "fit_status",
        "quality_status",
        "feature_profile_used",
        "elasticity_estimate",
        "lower_ci",
        "upper_ci",
        "sample_size",
        "non_null_pair_count",
        "unique_price_values",
        "log_price_std",
        "log_units_std",
        "feature_count_used",
        "skip_reason",
        "warning_count",
        "inference_warning_count",
        "warnings_text",
        "ci_caution_flag",
    ]

    estimates_path = paths.artifacts_dir / "elasticity_estimates.csv"
    if not estimates_path.exists():
        warnings.append("Elasticity estimates artifact is missing; elasticity summary will be empty")
        return pd.DataFrame(columns=columns)

    estimates = pd.read_csv(estimates_path)
    if estimates.empty:
        return pd.DataFrame(columns=columns)

    required = {"segment_key", "fit_status", "elasticity_estimate", "lower_ci", "upper_ci", "sample_size", "skip_reason"}
    missing = sorted(required - set(estimates.columns))
    if missing:
        raise ValueError("elasticity_estimates.csv is missing columns: " + ", ".join(missing))

    if "quality_status" not in estimates.columns:
        estimates["quality_status"] = _derive_elasticity_quality_status(estimates)

    if "warning_count" not in estimates.columns:
        estimates["warning_count"] = 0
    if "inference_warning_count" not in estimates.columns:
        estimates["inference_warning_count"] = 0
    if "warnings_text" not in estimates.columns:
        estimates["warnings_text"] = ""
    if "feature_profile_used" not in estimates.columns:
        estimates["feature_profile_used"] = "unknown"
    if "feature_count_used" not in estimates.columns:
        estimates["feature_count_used"] = 0
    if "non_null_pair_count" not in estimates.columns:
        estimates["non_null_pair_count"] = 0
    if "unique_price_values" not in estimates.columns:
        estimates["unique_price_values"] = 0
    if "log_price_std" not in estimates.columns:
        estimates["log_price_std"] = np.nan
    if "log_units_std" not in estimates.columns:
        estimates["log_units_std"] = np.nan

    estimates["warning_count"] = pd.to_numeric(estimates["warning_count"], errors="coerce").fillna(0).astype("int64")
    estimates["inference_warning_count"] = (
        pd.to_numeric(estimates["inference_warning_count"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    estimates["feature_count_used"] = (
        pd.to_numeric(estimates["feature_count_used"], errors="coerce").fillna(0).astype("int64")
    )
    estimates["non_null_pair_count"] = (
        pd.to_numeric(estimates["non_null_pair_count"], errors="coerce").fillna(0).astype("int64")
    )
    estimates["unique_price_values"] = (
        pd.to_numeric(estimates["unique_price_values"], errors="coerce").fillna(0).astype("int64")
    )
    estimates["log_price_std"] = pd.to_numeric(estimates["log_price_std"], errors="coerce").astype("float64")
    estimates["log_units_std"] = pd.to_numeric(estimates["log_units_std"], errors="coerce").astype("float64")
    estimates["feature_profile_used"] = estimates["feature_profile_used"].fillna("unknown").astype("string")
    estimates["warnings_text"] = estimates["warnings_text"].fillna("").astype("string")

    estimates["ci_caution_flag"] = (
        (estimates["quality_status"].astype("string") == "warning_inference_unstable")
        | (estimates["inference_warning_count"] > 0)
        | pd.to_numeric(estimates["lower_ci"], errors="coerce").isna()
        | pd.to_numeric(estimates["upper_ci"], errors="coerce").isna()
    )

    return estimates[columns].copy(deep=True)


def _derive_elasticity_quality_status(estimates: pd.DataFrame) -> pd.Series:
    fit_status = estimates["fit_status"].astype("string")
    lower_ci = pd.to_numeric(estimates["lower_ci"], errors="coerce")
    upper_ci = pd.to_numeric(estimates["upper_ci"], errors="coerce")
    skip_reason = estimates["skip_reason"].fillna("").astype("string").str.lower()
    warning_text = estimates.get("warnings_text", pd.Series("", index=estimates.index)).fillna("").astype("string").str.lower()
    inference_warning_count = pd.to_numeric(
        estimates.get("inference_warning_count", pd.Series(0, index=estimates.index)),
        errors="coerce",
    ).fillna(0)

    quality = pd.Series("ok", index=estimates.index, dtype="string")
    quality.loc[fit_status == "skipped"] = "skipped"
    quality.loc[fit_status == "failed"] = "failed"

    unstable_mask = (
        (fit_status == "success")
        & (
            lower_ci.isna()
            | upper_ci.isna()
            | (inference_warning_count > 0)
            | skip_reason.str.contains("covariance|inference|singular")
            | warning_text.str.contains("covariance|co-variance|inference|singular")
        )
    )
    quality.loc[unstable_mask] = "warning_inference_unstable"
    return quality


def _select_best_overall_row(
    forecast_summary: pd.DataFrame,
    best_registry: pd.DataFrame,
) -> pd.Series | None:
    if not forecast_summary.empty:
        return forecast_summary.iloc[0]

    if best_registry.empty:
        return None

    overall = best_registry.loc[best_registry["scope"] == "overall"].head(1)
    if overall.empty:
        return None

    row = overall.iloc[0]
    mapped = pd.Series(
        {
            "model_name": row.get("model_name"),
            "model_family": row.get("model_family"),
            "validation_mape": row.get("validation_mape"),
            "validation_wmape": row.get("validation_wmape"),
            "validation_mae": row.get("validation_mae"),
            "validation_rmse": row.get("validation_rmse"),
            "test_mape": row.get("test_mape"),
            "test_wmape": row.get("test_wmape"),
            "test_mae": row.get("test_mae"),
            "test_rmse": row.get("test_rmse"),
        }
    )
    return mapped


def _build_final_project_summary_payload(
    paths: ProjectPaths,
    forecast_summary: pd.DataFrame,
    best_overall: pd.Series | None,
    elasticity_summary: pd.DataFrame,
    llm_summary: dict[str, object] | None,
    llm_usage: LLMUsageFacts,
    elasticity_overview: dict[str, object],
    warnings: list[str],
) -> dict[str, object]:
    data_summary = _load_json_if_exists(paths.artifacts_dir / "data_summary.json")
    split_summary = _load_json_if_exists(paths.artifacts_dir / "split_summary.json")
    manual_feature_summary = _load_json_if_exists(paths.artifacts_dir / "features_manual_summary.json")

    best_model_name = str(best_overall["model_name"]) if best_overall is not None else "unknown"
    best_validation_wmape = _to_float(best_overall.get("validation_wmape") if best_overall is not None else np.nan)
    best_test_wmape = _to_float(best_overall.get("test_wmape") if best_overall is not None else np.nan)

    baseline_subset = forecast_summary.loc[
        forecast_summary["model_family"].astype("string") == "baseline"
    ].copy()
    baseline_best_wmape = float("nan")
    if not baseline_subset.empty:
        baseline_best_wmape = _to_float(baseline_subset["validation_wmape"].min())

    return {
        "generated_at_utc": _utc_now_iso(),
        "best_model_name": best_model_name,
        "best_validation_wmape": best_validation_wmape,
        "best_test_wmape": best_test_wmape,
        "best_baseline_validation_wmape": baseline_best_wmape,
        "forecast_model_count": int(len(forecast_summary)),
        "elasticity_segments": int(len(elasticity_summary)),
        "elasticity_quality_counts": {
            str(key): int(value)
            for key, value in _as_str_int_dict(
                elasticity_overview.get("quality_status_counts", {})
            ).items()
        },
        "elasticity_inference_warnings_present": bool(
            elasticity_overview.get("inference_warnings_present", False)
        ),
        "elasticity_inference_warning_count": _coerce_int(
            elasticity_overview.get("inference_warning_count", 0)
        ),
        "elasticity_ci_caution_present": bool(
            elasticity_overview.get("ci_caution_present", False)
        ),
        "llm_requested": bool(llm_usage.llm_requested),
        "ollama_reachable": llm_usage.ollama_reachable,
        "planner_model_available": llm_usage.planner_model_available,
        "llm_feature_file_exists": bool(llm_usage.llm_feature_file_exists),
        "llm_output_feature_count": int(llm_usage.llm_output_feature_count),
        "llm_features_actually_used": bool(llm_usage.llm_features_actually_used),
        "llm_feature_columns_used": (
            llm_usage.llm_feature_columns_used if llm_usage.llm_features_actually_used else []
        ),
        "llm_usage_facts": llm_usage_facts_to_dict(llm_usage),
        "data_summary": data_summary,
        "split_summary": split_summary,
        "features_manual_summary": manual_feature_summary,
        "llm_features_summary": llm_summary,
        "warnings": warnings,
    }


def _build_run_manifest_payload(
    paths: ProjectPaths,
    optimize_metric: str,
    best_overall: pd.Series | None,
    warnings: list[str],
    output_paths: Mapping[str, Path],
    llm_summary: dict[str, object] | None,
    llm_usage: LLMUsageFacts,
    elasticity_overview: Mapping[str, object],
    run_config_values: Mapping[str, object] | None,
    stage_records: Sequence[dict[str, object]] | None,
) -> dict[str, object]:
    best_model_name = str(best_overall["model_name"]) if best_overall is not None else "unknown"
    best_metric_value = (
        _to_float(best_overall.get(f"validation_{optimize_metric}"))
        if best_overall is not None
        else float("nan")
    )

    source_paths = {
        "data_summary_json": str(paths.artifacts_dir / "data_summary.json"),
        "split_summary_json": str(paths.artifacts_dir / "split_summary.json"),
        "features_manual_summary_json": str(paths.artifacts_dir / "features_manual_summary.json"),
        "llm_features_summary_json": str(paths.artifacts_dir / "llm_features_summary.json"),
        "elasticity_estimates_csv": str(paths.artifacts_dir / "elasticity_estimates.csv"),
        "elasticity_run_summary_json": str(paths.artifacts_dir / "elasticity_run_summary.json"),
        "forecast_metrics_validation_csv": str(paths.artifacts_dir / "forecast_metrics_validation.csv"),
        "forecast_metrics_test_csv": str(paths.artifacts_dir / "forecast_metrics_test.csv"),
        "best_model_registry_csv": str(paths.artifacts_dir / "best_model_registry.csv"),
    }

    return {
        "run_timestamp_utc": _utc_now_iso(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "selected_config_values": dict(run_config_values or {}),
        "source_artifact_paths": source_paths,
        "llm_requested": bool(llm_usage.llm_requested),
        "ollama_reachable": llm_usage.ollama_reachable,
        "planner_model_available": llm_usage.planner_model_available,
        "llm_feature_file_exists": bool(llm_usage.llm_feature_file_exists),
        "llm_output_feature_count": int(llm_usage.llm_output_feature_count),
        "llm_features_actually_used": bool(llm_usage.llm_features_actually_used),
        "llm_feature_columns_used": (
            llm_usage.llm_feature_columns_used if llm_usage.llm_features_actually_used else []
        ),
        "llm_usage_facts": llm_usage_facts_to_dict(llm_usage),
        "elasticity_inference_warnings_present": bool(
            elasticity_overview.get("inference_warnings_present", False)
        ),
        "elasticity_inference_warning_count": _coerce_int(
            elasticity_overview.get("inference_warning_count", 0)
        ),
        "elasticity_ci_caution_present": bool(
            elasticity_overview.get("ci_caution_present", False)
        ),
        "chosen_best_model": best_model_name,
        "best_validation_metric_name": optimize_metric,
        "best_validation_metric_value": best_metric_value,
        "warnings": warnings,
        "llm_features_summary": llm_summary,
        "stage_records": list(stage_records or []),
        "output_artifact_paths": {name: str(path) for name, path in output_paths.items()},
    }


def _render_final_markdown_report(
    paths: ProjectPaths,
    forecast_summary: pd.DataFrame,
    best_overall: pd.Series | None,
    elasticity_summary: pd.DataFrame,
    llm_summary: dict[str, object] | None,
    llm_usage: LLMUsageFacts,
    elasticity_overview: Mapping[str, object],
    warnings: Sequence[str],
    outputs: Mapping[str, Path],
) -> str:
    data_summary = _load_json_if_exists(paths.artifacts_dir / "data_summary.json") or {}
    split_summary = _load_json_if_exists(paths.artifacts_dir / "split_summary.json") or {}
    manual_summary = _load_json_if_exists(paths.artifacts_dir / "features_manual_summary.json") or {}
    llm_summary_payload = llm_summary or {}
    elasticity_run = _load_json_if_exists(paths.artifacts_dir / "elasticity_run_summary.json") or {}

    lines: list[str] = []
    lines.append("# Final Run Report")
    lines.append("")
    lines.append(f"Generated: {_utc_now_iso()}")
    lines.append("")

    lines.append("## Data Preparation")
    lines.append(f"- Source file: {data_summary.get('source_filename', 'unknown')}")
    lines.append(f"- Rows: {data_summary.get('total_row_count', 'unknown')}")
    lines.append(f"- Date range: {data_summary.get('min_date', 'unknown')} to {data_summary.get('max_date', 'unknown')}")
    cutoffs = split_summary.get("cutoffs") if isinstance(split_summary, dict) else {}
    if isinstance(cutoffs, dict):
        lines.append(f"- Validation start: {cutoffs.get('validation_start', 'unknown')}")
        lines.append(f"- Test start: {cutoffs.get('test_start', 'unknown')}")
    lines.append("")

    lines.append("## Feature Generation")
    lines.append(f"- Manual feature count: {manual_summary.get('feature_column_count', 'unknown')}")
    lines.append(f"- LLM accepted specs: {llm_summary_payload.get('accepted_spec_count', 'unknown')}")
    lines.append(f"- LLM requested: {llm_usage.llm_requested}")
    lines.append(f"- Ollama reachable: {llm_usage.ollama_reachable}")
    lines.append(f"- Planner model available: {llm_usage.planner_model_available}")
    lines.append(f"- LLM feature file exists: {llm_usage.llm_feature_file_exists}")
    lines.append(f"- LLM output feature count: {llm_usage.llm_output_feature_count}")
    lines.append(f"- LLM features actually used downstream: {llm_usage.llm_features_actually_used}")
    lines.append(
        "- LLM feature columns used: "
        + (", ".join(llm_usage.llm_feature_columns_used) if llm_usage.llm_features_actually_used else "None")
    )
    lines.append("")

    lines.append("## Elasticity Summary")
    lines.append(f"- Segments attempted: {elasticity_run.get('total_segments_attempted', 'unknown')}")
    lines.append(f"- Successful fits: {elasticity_run.get('successful_fits', 'unknown')}")
    lines.append(f"- Skipped fits: {elasticity_run.get('skipped_fits', 'unknown')}")
    lines.append(f"- Failed fits: {elasticity_run.get('failed_fits', 'unknown')}")
    lines.append(
        f"- Inference warnings present: {elasticity_overview.get('inference_warnings_present', 'unknown')}"
    )
    lines.append(
        f"- Inference warning count: {elasticity_overview.get('inference_warning_count', 'unknown')}"
    )
    lines.append(
        f"- CI caution present: {elasticity_overview.get('ci_caution_present', 'unknown')}"
    )
    quality_counts = _as_str_int_dict(elasticity_overview.get("quality_status_counts", {}))
    if quality_counts:
        lines.append("- Quality counts:")
        for key, value in quality_counts.items():
            lines.append(f"  - {key}: {int(value)}")
    lines.append("")

    lines.append("## Forecasting Model Comparison")
    if forecast_summary.empty:
        lines.append("- Forecast summary unavailable. Run evaluate-forecast-models first.")
    else:
        for row in forecast_summary.head(8).itertuples(index=False):
            lines.append(
                "- "
                f"{getattr(row, 'model_name')}: "
                f"validation wMAPE={getattr(row, 'validation_wmape'):.6f}, "
                f"test wMAPE={getattr(row, 'test_wmape'):.6f}"
            )

    lines.append("")
    lines.append("## Best Model")
    if best_overall is None:
        lines.append("- Best model unavailable.")
    else:
        lines.append(f"- Winner: {best_overall.get('model_name', 'unknown')}")
        lines.append(f"- Validation wMAPE: {_to_float(best_overall.get('validation_wmape'))}")
        lines.append(f"- Test wMAPE: {_to_float(best_overall.get('test_wmape'))}")

    lines.append("")
    lines.append("## Caveats")
    lines.append("- Dataset remains synthetic and should not be treated as calibrated production demand truth.")
    lines.append("- Zero-demand days can distort percentage metrics; wMAPE remains primary optimization metric.")
    lines.append("- Observational pricing limits causal interpretation confidence even with DML controls.")

    lines.append("")
    lines.append("## Artifact Locations")
    for name, path in outputs.items():
        lines.append(f"- {name}: {path}")

    if warnings:
        lines.append("")
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines).strip() + "\n"


def _build_acceptance_summary_payload(
    optimize_metric: str,
    best_overall: pd.Series | None,
    llm_usage: LLMUsageFacts,
    elasticity_overview: Mapping[str, object],
    output_paths: Mapping[str, Path],
) -> dict[str, object]:
    best_model_name = str(best_overall.get("model_name")) if best_overall is not None else "unknown"
    best_model_family = str(best_overall.get("model_family")) if best_overall is not None else "unknown"

    return {
        "generated_at_utc": _utc_now_iso(),
        "forecasting_winner": {
            "model_name": best_model_name,
            "model_family": best_model_family,
            "optimize_metric": optimize_metric,
        },
        "validation_metrics": {
            "mape": _to_float(best_overall.get("validation_mape") if best_overall is not None else np.nan),
            "wmape": _to_float(best_overall.get("validation_wmape") if best_overall is not None else np.nan),
            "mae": _to_float(best_overall.get("validation_mae") if best_overall is not None else np.nan),
            "rmse": _to_float(best_overall.get("validation_rmse") if best_overall is not None else np.nan),
        },
        "test_metrics": {
            "mape": _to_float(best_overall.get("test_mape") if best_overall is not None else np.nan),
            "wmape": _to_float(best_overall.get("test_wmape") if best_overall is not None else np.nan),
            "mae": _to_float(best_overall.get("test_mae") if best_overall is not None else np.nan),
            "rmse": _to_float(best_overall.get("test_rmse") if best_overall is not None else np.nan),
        },
        "elasticity_warning_presence": bool(
            elasticity_overview.get("inference_warnings_present", False)
        ),
        "elasticity_warning_count": _coerce_int(elasticity_overview.get("inference_warning_count", 0)),
        "llm_requested": bool(llm_usage.llm_requested),
        "ollama_reachable": llm_usage.ollama_reachable,
        "planner_model_available": llm_usage.planner_model_available,
        "llm_output_feature_count": int(llm_usage.llm_output_feature_count),
        "llm_features_actually_used": bool(llm_usage.llm_features_actually_used),
        "llm_feature_columns_used": (
            llm_usage.llm_feature_columns_used if llm_usage.llm_features_actually_used else []
        ),
        "key_artifact_paths": {name: str(path) for name, path in output_paths.items()},
    }


def _build_elasticity_overview(
    elasticity_summary: pd.DataFrame,
    project_paths: ProjectPaths,
) -> dict[str, object]:
    run_summary = _load_json_if_exists(project_paths.artifacts_dir / "elasticity_run_summary.json") or {}

    summary_warning_present = bool(_coerce_bool(run_summary.get("inference_warnings_present")) or False)
    summary_warning_count = _coerce_int(run_summary.get("inference_warning_count", 0))
    summary_ci_caution = bool(_coerce_bool(run_summary.get("ci_caution_present")) or False)

    if elasticity_summary.empty:
        warning_count = summary_warning_count
        if summary_warning_present and warning_count == 0:
            warning_count = 1
        return {
            "inference_warnings_present": bool(summary_warning_present or warning_count > 0),
            "inference_warning_count": int(warning_count),
            "ci_caution_present": bool(summary_ci_caution or warning_count > 0),
            "quality_status_counts": {
                str(key): int(value)
                for key, value in _as_str_int_dict(run_summary.get("quality_status_counts", {})).items()
            },
        }

    table_warning_count = int(
        pd.to_numeric(elasticity_summary["inference_warning_count"], errors="coerce").fillna(0).sum()
    )
    quality_warning_count = int(
        (elasticity_summary["quality_status"].astype("string") == "warning_inference_unstable").sum()
    )

    warning_count = max(summary_warning_count, table_warning_count, quality_warning_count)
    if summary_warning_present and warning_count == 0:
        warning_count = 1

    table_ci_caution = bool(
        pd.Series(elasticity_summary["ci_caution_flag"], dtype="boolean").fillna(False).any()
    )
    quality_counts = {
        str(key): int(value)
        for key, value in elasticity_summary["quality_status"].astype("string").value_counts().items()
    }

    return {
        "inference_warnings_present": bool(summary_warning_present or warning_count > 0),
        "inference_warning_count": int(warning_count),
        "ci_caution_present": bool(summary_ci_caution or table_ci_caution or warning_count > 0),
        "quality_status_counts": quality_counts,
    }


def _resolve_llm_requested(
    run_config_values: Mapping[str, object] | None,
    llm_summary: Mapping[str, object] | None,
) -> bool:
    if run_config_values is not None:
        for key in ("use_llm_features", "llm_requested", "llm_flag_requested"):
            coerced = _coerce_bool(run_config_values.get(key))
            if coerced is not None:
                return coerced

    if llm_summary is not None:
        for key in ("llm_requested", "llm_flag_requested", "materialization_enabled"):
            coerced = _coerce_bool(llm_summary.get(key))
            if coerced is not None:
                return coerced
        return True

    return False


def _extract_best_llm_feature_columns(best_registry: pd.DataFrame) -> str:
    if best_registry.empty:
        return ""

    overall = best_registry.loc[best_registry["scope"].astype("string") == "overall"].head(1)
    if overall.empty:
        return ""

    return _normalize_llm_columns(overall.iloc[0].get("llm_feature_columns", ""))


def _as_str_int_dict(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, int] = {}
    for key, raw_value in value.items():
        normalized[str(key)] = _coerce_int(raw_value, default=0)
    return normalized


def _resolve_report_output_path(paths: ProjectPaths, output_path: Path | None) -> Path:
    candidate = paths.project_root / "reports" / "final_run_report.md" if output_path is None else Path(output_path)
    if not candidate.is_absolute():
        candidate = paths.project_root / candidate
    return candidate


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = load_json(path)
    return payload if isinstance(payload, dict) else None


def _to_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return float("nan")
    return float("nan")


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


def _coerce_bool(value: object) -> bool | None:
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


def _normalize_llm_columns(value: object) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    if isinstance(value, str):
        normalized = value.strip()
        if normalized.lower() in {"nan", "none"}:
            return ""
        return normalized

    normalized = str(value).strip()
    if normalized.lower() in {"nan", "none"}:
        return ""
    return normalized


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()
