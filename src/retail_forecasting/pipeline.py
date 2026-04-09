"""Phase 7 end-to-end orchestration for local-first forecasting workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from collections.abc import Callable
from pathlib import Path
import platform
import sys

from retail_forecasting.causal_dml import ElasticityRunConfig, fit_elasticity_pipeline
from retail_forecasting.data_loading import discover_and_load_csv
from retail_forecasting.data_validation import validate_and_standardize_dataframe
from retail_forecasting.evaluation import (
    BaselineBenchmarkConfig,
    ForecastEvaluationConfig,
    evaluate_forecast_models_pipeline,
    run_baseline_benchmark_pipeline,
)
from retail_forecasting.features_llm import build_llm_features_pipeline, plan_llm_features_pipeline
from retail_forecasting.features_manual import build_manual_features_pipeline
from retail_forecasting.forecasting_models import ForecastModelTrainingConfig, train_forecast_models_pipeline
from retail_forecasting.paths import build_project_paths
from retail_forecasting.preprocessing import prepare_data_pipeline, write_json
from retail_forecasting.reporting import ReportingConfig, generate_reporting_artifacts
from retail_forecasting.tuning import ForecastTuningConfig, tune_forecast_models_pipeline


@dataclass(frozen=True, slots=True)
class FullPipelineConfig:
    """Configuration for full-phase orchestration and stage skipping behavior."""

    input_path: Path | None = None
    use_llm_features: bool = True
    skip_llm: bool = False
    skip_elasticity: bool = False
    skip_tuning: bool = True
    segment_level: str = "product"
    optimize_metric: str = "wmape"
    segment_mode: str = "global"
    model: str = "all"
    random_state: int = 42
    n_trials: int = 20
    force_refresh_report: bool = False


# Backwards-compatible alias for older imports.
PipelineConfig = FullPipelineConfig


def run_full_pipeline(config: FullPipelineConfig | None = None) -> dict[str, Path]:
    """Run major pipeline stages in order with explicit skip controls and manifest reporting."""
    run_config = config if config is not None else FullPipelineConfig()

    stage_records: list[dict[str, object]] = []
    warnings: list[str] = []
    outputs: dict[str, Path] = {}

    def run_stage(
        stage_name: str,
        func: Callable[[], dict[str, Path]],
        *,
        skip: bool = False,
        skip_reason: str = "",
    ) -> dict[str, Path]:
        started = _utc_now_iso()
        if skip:
            stage_records.append(
                {
                    "stage": stage_name,
                    "status": "skipped",
                    "started_at_utc": started,
                    "finished_at_utc": _utc_now_iso(),
                    "skip_reason": skip_reason,
                    "artifacts": {},
                }
            )
            return {}

        try:
            result = func()
            stage_records.append(
                {
                    "stage": stage_name,
                    "status": "success",
                    "started_at_utc": started,
                    "finished_at_utc": _utc_now_iso(),
                    "skip_reason": "",
                    "artifacts": {key: str(value) for key, value in result.items()},
                }
            )
            return result
        except Exception as exc:  # noqa: BLE001
            stage_records.append(
                {
                    "stage": stage_name,
                    "status": "failed",
                    "started_at_utc": started,
                    "finished_at_utc": _utc_now_iso(),
                    "skip_reason": "",
                    "error": str(exc),
                    "artifacts": {},
                }
            )
            raise

    try:
        run_stage("validate-data", lambda: _validate_data_stage(run_config.input_path))

        outputs.update(
            run_stage(
                "prepare-data",
                lambda: prepare_data_pipeline(input_path=run_config.input_path),
            )
        )

        outputs.update(
            run_stage(
                "build-manual-features",
                lambda: build_manual_features_pipeline(),
            )
        )

        outputs.update(
            run_stage(
                "plan-llm-features",
                lambda: plan_llm_features_pipeline(include_manual_input=True),
                skip=run_config.skip_llm,
                skip_reason="skip_llm flag is enabled",
            )
        )

        outputs.update(
            run_stage(
                "build-llm-features",
                lambda: build_llm_features_pipeline(include_manual_input=True),
                skip=run_config.skip_llm,
                skip_reason="skip_llm flag is enabled",
            )
        )

        outputs.update(
            run_stage(
                "fit-elasticity",
                lambda: fit_elasticity_pipeline(
                    ElasticityRunConfig(
                        segment_level=run_config.segment_level,
                        use_llm_features=run_config.use_llm_features,
                    )
                ),
                skip=run_config.skip_elasticity,
                skip_reason="skip_elasticity flag is enabled",
            )
        )

        outputs.update(
            run_stage(
                "run-baselines",
                lambda: run_baseline_benchmark_pipeline(
                    BaselineBenchmarkConfig(
                        use_llm_features=run_config.use_llm_features,
                        input_path=run_config.input_path,
                    )
                ),
            )
        )

        outputs.update(
            run_stage(
                "tune-forecast-models",
                lambda: tune_forecast_models_pipeline(
                    ForecastTuningConfig(
                        model=run_config.model,
                        optimize_metric=run_config.optimize_metric,
                        n_trials=run_config.n_trials,
                        random_state=run_config.random_state,
                        use_llm_features=run_config.use_llm_features,
                        input_path=run_config.input_path,
                    )
                ),
                skip=run_config.skip_tuning,
                skip_reason="skip_tuning flag is enabled",
            )
        )

        outputs.update(
            run_stage(
                "train-forecast-models",
                lambda: train_forecast_models_pipeline(
                    ForecastModelTrainingConfig(
                        model=run_config.model,
                        segment_mode=run_config.segment_mode,
                        random_state=run_config.random_state,
                        use_tuned_params=not run_config.skip_tuning,
                        use_llm_features=run_config.use_llm_features,
                        input_path=run_config.input_path,
                    )
                ),
            )
        )

        outputs.update(
            run_stage(
                "evaluate-forecast-models",
                lambda: evaluate_forecast_models_pipeline(
                    ForecastEvaluationConfig(
                        model=run_config.model,
                        optimize_metric=run_config.optimize_metric,
                        segment_mode=run_config.segment_mode,
                        use_llm_features=run_config.use_llm_features,
                        input_path=run_config.input_path,
                    )
                ),
            )
        )

        report_outputs = run_stage(
            "generate-report",
            lambda: generate_reporting_artifacts(
                ReportingConfig(
                    optimize_metric=run_config.optimize_metric,
                    force_refresh_report=run_config.force_refresh_report,
                    run_config_values=_config_as_json_dict(run_config),
                    stage_records=stage_records,
                    upstream_warnings=warnings,
                )
            ),
        )
        outputs.update(report_outputs)

        return outputs

    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Pipeline execution stopped at failed stage: {exc}")
        _write_failure_manifest(
            run_config=run_config,
            stage_records=stage_records,
            warnings=warnings,
        )
        raise RuntimeError(f"run-full-pipeline failed: {exc}") from exc


def run_pipeline(config: FullPipelineConfig | None = None) -> dict[str, Path]:
    """Compatibility wrapper around run_full_pipeline."""
    return run_full_pipeline(config=config)


def _validate_data_stage(input_path: Path | None) -> dict[str, Path]:
    source_path, raw_df = discover_and_load_csv(input_path=input_path)
    standardized_df, _ = validate_and_standardize_dataframe(raw_df)

    # Validate-only stage returns source path metadata while preserving Path-type contract.
    _ = standardized_df
    return {"source_csv": source_path}


def _write_failure_manifest(
    run_config: FullPipelineConfig,
    stage_records: list[dict[str, object]],
    warnings: list[str],
) -> Path:
    paths = build_project_paths()
    manifest_payload: dict[str, object] = {
        "run_timestamp_utc": _utc_now_iso(),
        "status": "failed",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "selected_config_values": _config_as_json_dict(run_config),
        "warnings": warnings,
        "stage_records": stage_records,
    }
    return write_json(manifest_payload, paths.artifacts_dir / "run_manifest.json")


def _config_as_json_dict(config: FullPipelineConfig) -> dict[str, object]:
    payload = asdict(config)
    normalized: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            normalized[key] = str(value)
        else:
            normalized[key] = value
    return normalized


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()
