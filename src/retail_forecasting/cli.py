"""Typer command line interface for repository utility operations."""

from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
import platform
import sys
from typing import Any

from rich.console import Console
from rich.table import Table
import typer

from retail_forecasting import __version__
from retail_forecasting.causal_dml import (
    ElasticityRunConfig,
    fit_elasticity_pipeline,
    load_elasticity_estimates,
    load_elasticity_run_summary,
)
from retail_forecasting.config import get_settings
from retail_forecasting.data_loading import discover_and_load_csv
from retail_forecasting.data_validation import DataValidationError, validate_and_standardize_dataframe
from retail_forecasting.evaluation import (
    BaselineBenchmarkConfig,
    ForecastEvaluationConfig,
    evaluate_forecast_models_pipeline,
    load_best_model_registry,
    load_forecast_metrics,
    run_baseline_benchmark_pipeline,
)
from retail_forecasting.features_llm import (
    build_llm_features_pipeline,
    load_llm_features_summary,
    plan_llm_features_pipeline,
)
from retail_forecasting.features_manual import build_manual_features_pipeline, load_features_summary
from retail_forecasting.forecasting_models import (
    ForecastModelTrainingConfig,
    load_model_training_registry,
    train_forecast_models_pipeline,
)
from retail_forecasting.logging_utils import configure_logging
from retail_forecasting.paths import ProjectPaths, build_project_paths
from retail_forecasting.pipeline import FullPipelineConfig, run_full_pipeline
from retail_forecasting.predict import (
    ForecastExportConfig,
    ForecastNextConfig,
    export_forecasts_pipeline,
    forecast_next_pipeline,
)
from retail_forecasting.preprocessing import load_json, prepare_data_pipeline
from retail_forecasting.reporting import ReportingConfig, generate_reporting_artifacts
from retail_forecasting.tuning import ForecastTuningConfig, tune_forecast_models_pipeline
from retail_forecasting.utils import ensure_directory

app = typer.Typer(add_completion=False, help="Retail forecasting engine utilities.")
console = Console()


def _as_dict(value: object) -> dict[str, Any]:
    """Convert unknown objects to dictionary values where possible.

    Args:
        value: Unknown runtime object.

    Returns:
        Dictionary representation or empty dictionary.
    """
    return value if isinstance(value, dict) else {}


def _as_str_list(value: object) -> list[str]:
    """Convert unknown objects to a list of strings.

    Args:
        value: Unknown runtime object.

    Returns:
        String list; non-list values return an empty list.
    """
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _as_dict_list(value: object) -> list[dict[str, Any]]:
    """Convert unknown objects to a list of dictionaries.

    Args:
        value: Unknown runtime object.

    Returns:
        List of dictionary values.
    """
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _build_paths_table(paths: ProjectPaths) -> Table:
    """Build a rich table for resolved repository paths.

    Args:
        paths: Resolved project paths.

    Returns:
        Rich table object.
    """
    table = Table(title="Resolved Paths", show_lines=True)
    table.add_column("Key", style="bold cyan")
    table.add_column("Path", style="green")

    for item in fields(paths):
        key = item.name
        value = getattr(paths, key)
        table.add_row(key, str(value))

    return table


@app.callback()
def app_callback(
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        help="Optional log level override (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    ),
) -> None:
    """Initialize CLI context."""
    configure_logging(log_level=log_level)


@app.command()
def info() -> None:
    """Display package, runtime, and model configuration details."""
    settings = get_settings()
    console.print("package: retail-forecasting-engine")
    console.print(f"version: {__version__}")
    console.print(f"python: {sys.version.split()[0]}")
    console.print(f"platform: {platform.platform()}")
    console.print(f"ollama_host: {settings.ollama_host}")
    console.print(f"ollama_model: {settings.ollama_model}")


@app.command("check-env")
def check_env(
    create_missing: bool = typer.Option(
        False,
        "--create-missing",
        help="Create missing core directories when set.",
    )
) -> None:
    """Validate critical environment and repository directories."""
    paths = build_project_paths()
    required_paths = [paths.data_raw_dir, paths.data_processed_dir, paths.artifacts_dir]

    missing_paths = [path for path in required_paths if not path.exists()]
    if create_missing:
        for path in missing_paths:
            ensure_directory(path)
        missing_paths = [path for path in required_paths if not path.exists()]

    status = Table(title="Environment Check", show_lines=True)
    status.add_column("Item", style="bold cyan")
    status.add_column("Value", style="green")

    settings = get_settings()
    status.add_row("OLLAMA_HOST", settings.ollama_host)
    status.add_row("OLLAMA_MODEL", settings.ollama_model)
    status.add_row("LOG_LEVEL", settings.log_level)
    status.add_row("Missing required dirs", str(len(missing_paths)))
    console.print(status)

    if missing_paths:
        for missing in missing_paths:
            console.print(f"missing: {missing}", style="red")
        raise typer.Exit(code=1)

    console.print("Environment check passed.", style="bold green")


@app.command("show-paths")
def show_paths() -> None:
    """Show all resolved repository paths."""
    paths = build_project_paths()
    console.print(_build_paths_table(paths))


@app.command("validate-data")
def validate_data(
    input_path: Path | None = typer.Option(
        None,
        "--input-path",
        help="Optional explicit CSV path. Defaults to auto-discovery in data/raw.",
    )
) -> None:
    """Validate schema and types of the raw retail dataset."""
    try:
        source_path, raw_df = discover_and_load_csv(input_path=input_path)
        standardized_df, optional_columns = validate_and_standardize_dataframe(raw_df)
    except (DataValidationError, FileNotFoundError, ValueError) as exc:
        console.print(f"Validation failed: {exc}", style="bold red")
        raise typer.Exit(code=1) from exc

    result = Table(title="Validation Result", show_lines=True)
    result.add_column("Metric", style="bold cyan")
    result.add_column("Value", style="green")
    result.add_row("Source", str(source_path))
    result.add_row("Rows", str(int(standardized_df.shape[0])))
    result.add_row("Columns", str(int(standardized_df.shape[1])))
    result.add_row("Optional columns detected", ", ".join(optional_columns) or "None")
    result.add_row(
        "Date range",
        f"{standardized_df['date'].min().date()} to {standardized_df['date'].max().date()}",
    )
    console.print(result)
    console.print("Raw dataset validation passed.", style="bold green")


@app.command("prepare-data")
def prepare_data(
    input_path: Path | None = typer.Option(
        None,
        "--input-path",
        help="Optional explicit CSV path. Defaults to auto-discovery in data/raw.",
    ),
    train_ratio: float = typer.Option(0.70, help="Chronological train split ratio."),
    validation_ratio: float = typer.Option(0.15, help="Chronological validation split ratio."),
    test_ratio: float = typer.Option(0.15, help="Chronological test split ratio."),
    validation_start: str | None = typer.Option(
        None,
        "--validation-start",
        help="Optional explicit validation start date (YYYY-MM-DD).",
    ),
    test_start: str | None = typer.Option(
        None,
        "--test-start",
        help="Optional explicit test start date (YYYY-MM-DD).",
    ),
) -> None:
    """Run ingestion, cleaning, summary generation, and chronological splitting."""
    try:
        outputs = prepare_data_pipeline(
            input_path=input_path,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            validation_start=validation_start,
            test_start=test_start,
        )
    except (DataValidationError, FileNotFoundError, ValueError) as exc:
        console.print(f"Prepare-data failed: {exc}", style="bold red")
        raise typer.Exit(code=1) from exc

    result = Table(title="Prepare Data Outputs", show_lines=True)
    result.add_column("Artifact", style="bold cyan")
    result.add_column("Path", style="green")
    for key, value in outputs.items():
        result.add_row(key, str(value))
    console.print(result)
    console.print("Data preparation completed.", style="bold green")


@app.command("summarize-data")
def summarize_data() -> None:
    """Print readable summaries from generated data artifacts."""
    paths = build_project_paths()
    data_summary_path = paths.artifacts_dir / "data_summary.json"
    split_summary_path = paths.artifacts_dir / "split_summary.json"

    if not data_summary_path.exists():
        console.print(
            f"Missing {data_summary_path}. Run 'prepare-data' first.",
            style="bold red",
        )
        raise typer.Exit(code=1)

    data_summary = load_json(data_summary_path)
    data_table = Table(title="Data Summary", show_lines=True)
    data_table.add_column("Field", style="bold cyan")
    data_table.add_column("Value", style="green")
    data_table.add_row("Source filename", str(data_summary.get("source_filename")))
    data_table.add_row("Total rows", str(data_summary.get("total_row_count")))
    data_table.add_row("Total columns", str(data_summary.get("total_column_count")))
    data_table.add_row(
        "Date range",
        f"{data_summary.get('min_date')} to {data_summary.get('max_date')}",
    )
    data_table.add_row("Stores", str(data_summary.get("number_of_stores")))
    data_table.add_row("Products", str(data_summary.get("number_of_products")))
    optional_columns = _as_str_list(data_summary.get("optional_columns_detected"))
    data_table.add_row("Optional columns", ", ".join(optional_columns) or "None")
    data_table.add_row(
        "Duplicate rows removed",
        str(data_summary.get("duplicate_row_count_removed")),
    )
    data_table.add_row("Zero-demand rate", str(data_summary.get("zero_demand_rate")))
    console.print(data_table)

    if split_summary_path.exists():
        split_summary = load_json(split_summary_path)
        split_map = _as_dict(split_summary.get("splits"))
        split_table = Table(title="Split Summary", show_lines=True)
        split_table.add_column("Split", style="bold cyan")
        split_table.add_column("Rows", style="green")
        split_table.add_column("Share", style="green")
        split_table.add_column("Date Range", style="green")

        for split_name in ("train", "validation", "test"):
            split_metrics = _as_dict(split_map.get(split_name))
            date_range = f"{split_metrics.get('min_date')} to {split_metrics.get('max_date')}"
            split_table.add_row(
                split_name,
                str(split_metrics.get("row_count")),
                str(split_metrics.get("share_of_total_rows")),
                date_range,
            )
        console.print(split_table)


@app.command("build-manual-features")
def build_manual_features(
    input_path: Path | None = typer.Option(
        None,
        "--input-path",
        help="Optional cleaned parquet path. Defaults to data/interim/cleaned_retail.parquet.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output-path",
        help="Optional output parquet path. Defaults to data/processed/features_manual.parquet.",
    ),
    keep_warmup_rows: bool = typer.Option(
        False,
        "--keep-warmup-rows",
        help="Keep rows without full lag/rolling history.",
    ),
    write_split_artifacts: bool = typer.Option(
        True,
        "--write-split-artifacts/--skip-split-artifacts",
        help="Write split-aware feature parquet artifacts when split files are available.",
    ),
) -> None:
    """Build leakage-safe manual features and persist artifacts."""
    try:
        outputs = build_manual_features_pipeline(
            input_path=input_path,
            output_path=output_path,
            drop_warmup_rows=not keep_warmup_rows,
            write_split_artifacts=write_split_artifacts,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Manual feature build failed: {exc}", style="bold red")
        raise typer.Exit(code=1) from exc

    result = Table(title="Manual Feature Artifacts", show_lines=True)
    result.add_column("Artifact", style="bold cyan")
    result.add_column("Path", style="green")
    for key, value in outputs.items():
        result.add_row(key, str(value))

    console.print(result)
    console.print("Manual feature engineering completed.", style="bold green")


@app.command("summarize-features")
def summarize_features() -> None:
    """Display a readable summary of the manual feature artifact metadata."""
    try:
        summary = load_features_summary()
    except FileNotFoundError as exc:
        console.print(str(exc), style="bold red")
        raise typer.Exit(code=1) from exc

    summary_table = Table(title="Manual Feature Summary", show_lines=True)
    summary_table.add_column("Field", style="bold cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Source input", str(summary.get("source_input_path")))
    summary_table.add_row(
        "Rows before feature generation",
        str(summary.get("row_count_before_feature_generation")),
    )
    summary_table.add_row(
        "Rows after warmup trimming",
        str(summary.get("row_count_after_warmup_trimming")),
    )
    summary_table.add_row("Rows dropped for warmup", str(summary.get("rows_dropped_for_warmup")))
    summary_table.add_row("Feature count", str(summary.get("feature_column_count")))
    summary_table.add_row(
        "Optional groups used",
        ", ".join(_as_str_list(summary.get("optional_feature_groups_used"))) or "None",
    )
    summary_table.add_row(
        "Missing optional columns",
        ", ".join(_as_str_list(summary.get("columns_skipped_missing"))) or "None",
    )
    summary_table.add_row("Leakage safety", str(summary.get("leakage_safety_note")))
    summary_table.add_row("Forecast hint", str(summary.get("forecast_error_hint_note")))
    console.print(summary_table)

    outputs = _as_dict(summary.get("output_paths"))
    if outputs:
        outputs_table = Table(title="Feature Output Paths", show_lines=True)
        outputs_table.add_column("Artifact", style="bold cyan")
        outputs_table.add_column("Path", style="green")
        for key in sorted(outputs):
            outputs_table.add_row(str(key), str(outputs[key]))
        console.print(outputs_table)


@app.command("plan-llm-features")
def plan_llm_features(
    input_path: Path | None = typer.Option(
        None,
        "--input-path",
        help="Optional input parquet path for planning context.",
    ),
    include_manual_input: bool = typer.Option(
        False,
        "--include-manual-input",
        help="Use data/processed/features_manual.parquet by default when available.",
    ),
) -> None:
    """Generate and validate LLM feature plans without materializing features."""
    try:
        outputs = plan_llm_features_pipeline(
            input_path=input_path,
            include_manual_input=include_manual_input,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"LLM feature planning failed: {exc}", style="bold red")
        raise typer.Exit(code=1) from exc

    result = Table(title="LLM Feature Planning Artifacts", show_lines=True)
    result.add_column("Artifact", style="bold cyan")
    result.add_column("Path", style="green")
    for key, value in outputs.items():
        result.add_row(key, str(value))
    console.print(result)

    console.print("LLM feature planning completed.", style="bold green")


@app.command("build-llm-features")
def build_llm_features(
    input_path: Path | None = typer.Option(
        None,
        "--input-path",
        help="Optional input parquet path for planning and materialization.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output-path",
        help="Optional output parquet path for materialized LLM features.",
    ),
    include_manual_input: bool = typer.Option(
        False,
        "--include-manual-input",
        help="Use data/processed/features_manual.parquet by default when available.",
    ),
) -> None:
    """Generate, validate, and materialize deterministic LLM-based features."""
    try:
        outputs = build_llm_features_pipeline(
            input_path=input_path,
            output_path=output_path,
            include_manual_input=include_manual_input,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"LLM feature build failed: {exc}", style="bold red")
        raise typer.Exit(code=1) from exc

    result = Table(title="LLM Feature Build Artifacts", show_lines=True)
    result.add_column("Artifact", style="bold cyan")
    result.add_column("Path", style="green")
    for key, value in outputs.items():
        result.add_row(key, str(value))
    console.print(result)

    console.print("LLM feature materialization completed.", style="bold green")


@app.command("summarize-llm-features")
def summarize_llm_features() -> None:
    """Display summary details for latest LLM planning/materialization artifacts."""
    try:
        summary = load_llm_features_summary()
    except FileNotFoundError as exc:
        console.print(str(exc), style="bold red")
        raise typer.Exit(code=1) from exc

    summary_table = Table(title="LLM Feature Summary", show_lines=True)
    summary_table.add_column("Field", style="bold cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Planner host", str(summary.get("planner_host_used")))
    summary_table.add_row("Planner model", str(summary.get("planner_model_used")))
    summary_table.add_row("Source dataset", str(summary.get("source_dataset_path")))
    summary_table.add_row("Ollama reachable", str(summary.get("ollama_reachable")))
    summary_table.add_row("Planner error", str(summary.get("planner_error") or "None"))
    summary_table.add_row("Raw spec count", str(summary.get("raw_spec_count")))
    summary_table.add_row("Accepted spec count", str(summary.get("accepted_spec_count")))
    summary_table.add_row("Rejected spec count", str(summary.get("rejected_spec_count")))
    summary_table.add_row("Output feature count", str(summary.get("output_feature_count")))
    summary_table.add_row(
        "Overlap with manual features",
        ", ".join(_as_str_list(summary.get("overlap_with_manual_features"))) or "None",
    )
    summary_table.add_row("Leakage safety", str(summary.get("leakage_safety_statement")))
    console.print(summary_table)

    rejected_specs = _as_dict_list(summary.get("rejected_specs"))
    if rejected_specs:
        rejection_table = Table(title="Rejected LLM Specs", show_lines=True)
        rejection_table.add_column("Feature", style="bold cyan")
        rejection_table.add_column("Operation", style="green")
        rejection_table.add_column("Reason", style="green")

        for item in rejected_specs:
            rejection_table.add_row(
                str(item.get("feature_name", "unknown")),
                str(item.get("operation", "unknown")),
                str(item.get("reason", "unknown")),
            )
        console.print(rejection_table)

    outputs = _as_dict(summary.get("output_paths"))
    if outputs:
        outputs_table = Table(title="LLM Output Paths", show_lines=True)
        outputs_table.add_column("Artifact", style="bold cyan")
        outputs_table.add_column("Path", style="green")
        for key in sorted(outputs):
            outputs_table.add_row(str(key), str(outputs[key]))
        console.print(outputs_table)


@app.command("fit-elasticity")
def fit_elasticity(
    segment_level: str = typer.Option(
        "product",
        "--segment-level",
        help="Segmentation level: product or store-product.",
    ),
    feature_profile: str = typer.Option(
        "lean",
        "--feature-profile",
        help="Causal control profile: lean or full.",
    ),
    min_samples: int = typer.Option(
        250,
        "--min-samples",
        help="Minimum rows required per segment.",
    ),
    min_non_null_pairs: int | None = typer.Option(
        None,
        "--min-non-null-pairs",
        help="Minimum non-null log(outcome)/log(treatment) pairs required per segment.",
    ),
    min_unique_price_values: int = typer.Option(
        8,
        "--min-unique-price-values",
        help="Minimum unique log(price) values required per segment.",
    ),
    min_log_price_std: float = typer.Option(
        0.01,
        "--min-log-price-std",
        help="Minimum standard deviation required for log(price).",
    ),
    min_log_units_std: float = typer.Option(
        0.01,
        "--min-log-units-std",
        help="Minimum standard deviation required for log(units_sold + epsilon).",
    ),
    epsilon: float = typer.Option(
        1e-3,
        "--epsilon",
        help="Small positive constant for safe log transforms.",
    ),
    use_llm_features: bool = typer.Option(
        True,
        "--use-llm-features/--no-use-llm-features",
        help=(
            "Load optional LLM feature artifacts. The default lean causal profile still excludes "
            "LLM-derived controls unless explicitly enabled in config."
        ),
    ),
    nuisance_model: str = typer.Option(
        "random-forest",
        "--nuisance-model",
        help="Nuisance model family: random-forest or gradient-boosting.",
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input-path",
        help="Optional override for primary causal input parquet.",
    ),
    llm_features_path: Path | None = typer.Option(
        None,
        "--llm-features-path",
        help="Optional override for LLM feature parquet path.",
    ),
) -> None:
    """Run segmented EconML LinearDML own-price elasticity estimation."""
    try:
        outputs = fit_elasticity_pipeline(
            ElasticityRunConfig(
                segment_level=segment_level,
                feature_profile=feature_profile,
                min_samples=min_samples,
                min_non_null_pairs=min_non_null_pairs,
                min_unique_price_values=min_unique_price_values,
                min_log_price_std=min_log_price_std,
                min_log_units_std=min_log_units_std,
                epsilon=epsilon,
                use_llm_features=use_llm_features,
                nuisance_model=nuisance_model,
                input_path=input_path,
                llm_features_path=llm_features_path,
            )
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        console.print(f"Elasticity fit failed: {exc}", style="bold red")
        raise typer.Exit(code=1) from exc

    result = Table(title="Elasticity Fit Artifacts", show_lines=True)
    result.add_column("Artifact", style="bold cyan")
    result.add_column("Path", style="green")
    for key, value in outputs.items():
        result.add_row(key, str(value))

    console.print(result)
    console.print("Causal elasticity estimation completed.", style="bold green")


@app.command("summarize-elasticity")
def summarize_elasticity() -> None:
    """Display summary and notable segments from latest elasticity artifacts."""
    try:
        summary = load_elasticity_run_summary()
        estimates = load_elasticity_estimates()
    except FileNotFoundError as exc:
        console.print(str(exc), style="bold red")
        raise typer.Exit(code=1) from exc

    summary_table = Table(title="Elasticity Run Summary", show_lines=True)
    summary_table.add_column("Field", style="bold cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Source data", str(summary.get("source_data_path")))
    summary_table.add_row("LLM augmentation used", str(summary.get("llm_feature_augmentation_used")))
    summary_table.add_row("Segmentation level", str(summary.get("segmentation_level")))
    summary_table.add_row("Feature profile", str(summary.get("feature_profile_used")))
    summary_table.add_row("Segments attempted", str(summary.get("total_segments_attempted")))
    summary_table.add_row("Successful fits", str(summary.get("successful_fits")))
    summary_table.add_row("Skipped fits", str(summary.get("skipped_fits")))
    summary_table.add_row("Inference warnings", str(summary.get("inference_warning_count")))
    summary_table.add_row("Inference warnings present", str(summary.get("inference_warnings_present")))
    summary_table.add_row("Min samples", str(summary.get("min_sample_threshold")))
    summary_table.add_row("Epsilon", str(summary.get("epsilon_used")))
    summary_table.add_row("Nuisance model", str(summary.get("nuisance_model_choice")))
    console.print(summary_table)

    successful = estimates.loc[
        (estimates.get("fit_status") == "success") & estimates.get("elasticity_estimate").notna()
    ].copy()

    if not successful.empty:
        top_most_elastic = successful.nsmallest(5, columns=["elasticity_estimate"])
        top_least_elastic = successful.nlargest(5, columns=["elasticity_estimate"])

        most_elastic_table = Table(title="Most Elastic Segments", show_lines=True)
        most_elastic_table.add_column("Segment", style="bold cyan")
        most_elastic_table.add_column("Estimate", style="green")
        most_elastic_table.add_column("95% CI", style="green")
        for row in top_most_elastic.itertuples(index=False):
            ci_text = f"[{getattr(row, 'lower_ci')}, {getattr(row, 'upper_ci')}]"
            most_elastic_table.add_row(str(getattr(row, "segment_key")), str(getattr(row, "elasticity_estimate")), ci_text)
        console.print(most_elastic_table)

        least_elastic_table = Table(title="Least Elastic Segments", show_lines=True)
        least_elastic_table.add_column("Segment", style="bold cyan")
        least_elastic_table.add_column("Estimate", style="green")
        least_elastic_table.add_column("95% CI", style="green")
        for row in top_least_elastic.itertuples(index=False):
            ci_text = f"[{getattr(row, 'lower_ci')}, {getattr(row, 'upper_ci')}]"
            least_elastic_table.add_row(str(getattr(row, "segment_key")), str(getattr(row, "elasticity_estimate")), ci_text)
        console.print(least_elastic_table)

    skipped = estimates.loc[estimates.get("fit_status") != "success"].copy()
    if not skipped.empty:
        reason_counts = skipped["skip_reason"].fillna("unknown").astype("string").value_counts()
        skip_table = Table(title="Skip Reasons", show_lines=True)
        skip_table.add_column("Reason", style="bold cyan")
        skip_table.add_column("Count", style="green")
        for reason, count in reason_counts.items():
            skip_table.add_row(str(reason), str(int(count)))
        console.print(skip_table)


@app.command("run-baselines")
def run_baselines(
    use_llm_features: bool = typer.Option(
        True,
        "--use-llm-features/--no-use-llm-features",
        help="Optionally merge non-overlapping LLM features when available.",
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input-path",
        help="Optional override for primary manual feature parquet input.",
    ),
    llm_features_path: Path | None = typer.Option(
        None,
        "--llm-features-path",
        help="Optional override for LLM feature parquet path.",
    ),
) -> None:
    """Compute baseline forecasts and persist baseline prediction/metric artifacts."""
    try:
        outputs = run_baseline_benchmark_pipeline(
            BaselineBenchmarkConfig(
                use_llm_features=use_llm_features,
                input_path=input_path,
                llm_features_path=llm_features_path,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Baseline run failed: {exc}", style="bold red")
        raise typer.Exit(code=1) from exc

    result = Table(title="Baseline Artifacts", show_lines=True)
    result.add_column("Artifact", style="bold cyan")
    result.add_column("Path", style="green")
    for key, value in outputs.items():
        result.add_row(key, str(value))
    console.print(result)
    console.print("Baseline benchmarking completed.", style="bold green")


@app.command("train-forecast-models")
def train_forecast_models(
    model: str = typer.Option(
        "all",
        "--model",
        help="Model family to train: lightgbm, xgboost, or all.",
    ),
    segment_mode: str = typer.Option(
        "global",
        "--segment-mode",
        help="Training mode: global or per-product.",
    ),
    random_state: int = typer.Option(
        42,
        "--random-state",
        help="Random seed for deterministic model fitting.",
    ),
    use_tuned_params: bool = typer.Option(
        True,
        "--use-tuned-params/--ignore-tuned-params",
        help="Use Optuna best params artifacts when available.",
    ),
    use_llm_features: bool = typer.Option(
        True,
        "--use-llm-features/--no-use-llm-features",
        help="Optionally merge non-overlapping LLM features when available.",
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input-path",
        help="Optional override for primary manual feature parquet input.",
    ),
    llm_features_path: Path | None = typer.Option(
        None,
        "--llm-features-path",
        help="Optional override for LLM feature parquet path.",
    ),
) -> None:
    """Train LightGBM/XGBoost forecast models and persist joblib artifacts."""
    try:
        outputs = train_forecast_models_pipeline(
            ForecastModelTrainingConfig(
                model=model,
                segment_mode=segment_mode,
                random_state=random_state,
                use_tuned_params=use_tuned_params,
                use_llm_features=use_llm_features,
                input_path=input_path,
                llm_features_path=llm_features_path,
            )
        )
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        console.print(f"Model training failed: {exc}", style="bold red")
        raise typer.Exit(code=1) from exc

    result = Table(title="Forecast Training Artifacts", show_lines=True)
    result.add_column("Artifact", style="bold cyan")
    result.add_column("Path", style="green")
    for key, value in outputs.items():
        result.add_row(key, str(value))
    console.print(result)

    try:
        training_registry = load_model_training_registry()
        status_table = Table(title="Training Status", show_lines=True)
        status_table.add_column("Model", style="bold cyan")
        status_table.add_column("Mode", style="green")
        status_table.add_column("Status", style="green")
        status_table.add_column("Feature Count", style="green")

        for row in training_registry.itertuples(index=False):
            status_table.add_row(
                str(getattr(row, "model_name", "")),
                str(getattr(row, "segment_mode", "")),
                str(getattr(row, "status", "")),
                str(getattr(row, "feature_count", "")),
            )
        console.print(status_table)
    except FileNotFoundError:
        pass

    console.print("Forecast model training completed.", style="bold green")


@app.command("tune-forecast-models")
def tune_forecast_models(
    model: str = typer.Option(
        "all",
        "--model",
        help="Model family to tune: lightgbm, xgboost, or all.",
    ),
    optimize_metric: str = typer.Option(
        "wmape",
        "--optimize-metric",
        help="Optimization metric: wmape, mape, mae, or rmse.",
    ),
    n_trials: int = typer.Option(
        20,
        "--n-trials",
        help="Optuna trial count per model family.",
    ),
    random_state: int = typer.Option(
        42,
        "--random-state",
        help="Random seed for deterministic tuning.",
    ),
    use_llm_features: bool = typer.Option(
        True,
        "--use-llm-features/--no-use-llm-features",
        help="Optionally merge non-overlapping LLM features when available.",
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input-path",
        help="Optional override for primary manual feature parquet input.",
    ),
    llm_features_path: Path | None = typer.Option(
        None,
        "--llm-features-path",
        help="Optional override for LLM feature parquet path.",
    ),
) -> None:
    """Run optional Optuna tuning for LightGBM/XGBoost and persist best params."""
    try:
        outputs = tune_forecast_models_pipeline(
            ForecastTuningConfig(
                model=model,
                optimize_metric=optimize_metric,
                n_trials=n_trials,
                random_state=random_state,
                use_llm_features=use_llm_features,
                input_path=input_path,
                llm_features_path=llm_features_path,
            )
        )
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        console.print(f"Model tuning failed: {exc}", style="bold red")
        raise typer.Exit(code=1) from exc

    result = Table(title="Forecast Tuning Artifacts", show_lines=True)
    result.add_column("Artifact", style="bold cyan")
    result.add_column("Path", style="green")
    for key, value in outputs.items():
        result.add_row(key, str(value))
    console.print(result)
    console.print("Forecast model tuning completed.", style="bold green")


@app.command("evaluate-forecast-models")
def evaluate_forecast_models(
    model: str = typer.Option(
        "all",
        "--model",
        help="Model family to evaluate: lightgbm, xgboost, or all.",
    ),
    optimize_metric: str = typer.Option(
        "wmape",
        "--optimize-metric",
        help="Selection metric: wmape, mape, mae, or rmse.",
    ),
    segment_mode: str = typer.Option(
        "global",
        "--segment-mode",
        help="Segment mode used for trained artifacts: global or per-product.",
    ),
    use_llm_features: bool = typer.Option(
        True,
        "--use-llm-features/--no-use-llm-features",
        help="Optionally merge non-overlapping LLM features when available.",
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input-path",
        help="Optional override for primary manual feature parquet input.",
    ),
    llm_features_path: Path | None = typer.Option(
        None,
        "--llm-features-path",
        help="Optional override for LLM feature parquet path.",
    ),
) -> None:
    """Evaluate baselines and trained models on validation/test splits."""
    try:
        outputs = evaluate_forecast_models_pipeline(
            ForecastEvaluationConfig(
                model=model,
                optimize_metric=optimize_metric,
                segment_mode=segment_mode,
                use_llm_features=use_llm_features,
                input_path=input_path,
                llm_features_path=llm_features_path,
            )
        )
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        console.print(f"Forecast evaluation failed: {exc}", style="bold red")
        raise typer.Exit(code=1) from exc

    result = Table(title="Forecast Evaluation Artifacts", show_lines=True)
    result.add_column("Artifact", style="bold cyan")
    result.add_column("Path", style="green")
    for key, value in outputs.items():
        result.add_row(key, str(value))
    console.print(result)
    console.print("Forecast evaluation completed.", style="bold green")


@app.command("summarize-forecasting")
def summarize_forecasting() -> None:
    """Print readable summary tables for latest forecasting model comparisons."""
    try:
        validation_metrics = load_forecast_metrics("validation")
        test_metrics = load_forecast_metrics("test")
        best_registry = load_best_model_registry()
    except FileNotFoundError as exc:
        console.print(str(exc), style="bold red")
        raise typer.Exit(code=1) from exc

    validation_table = Table(title="Validation Metrics", show_lines=True)
    validation_table.add_column("Model", style="bold cyan")
    validation_table.add_column("Family", style="green")
    validation_table.add_column("wMAPE", style="green")
    validation_table.add_column("MAPE", style="green")
    validation_table.add_column("MAE", style="green")
    validation_table.add_column("RMSE", style="green")

    for row in validation_metrics.sort_values(by=["wmape"], kind="mergesort").itertuples(index=False):
        validation_table.add_row(
            str(getattr(row, "model_name", "")),
            str(getattr(row, "model_family", "")),
            str(getattr(row, "wmape", "")),
            str(getattr(row, "mape", "")),
            str(getattr(row, "mae", "")),
            str(getattr(row, "rmse", "")),
        )
    console.print(validation_table)

    test_table = Table(title="Test Metrics", show_lines=True)
    test_table.add_column("Model", style="bold cyan")
    test_table.add_column("Family", style="green")
    test_table.add_column("wMAPE", style="green")
    test_table.add_column("MAPE", style="green")
    test_table.add_column("MAE", style="green")
    test_table.add_column("RMSE", style="green")

    for row in test_metrics.sort_values(by=["wmape"], kind="mergesort").itertuples(index=False):
        test_table.add_row(
            str(getattr(row, "model_name", "")),
            str(getattr(row, "model_family", "")),
            str(getattr(row, "wmape", "")),
            str(getattr(row, "mape", "")),
            str(getattr(row, "mae", "")),
            str(getattr(row, "rmse", "")),
        )
    console.print(test_table)

    overall_best = best_registry.loc[best_registry["scope"] == "overall"].head(1)
    if not overall_best.empty:
        best_row = overall_best.iloc[0]
        best_table = Table(title="Current Best Model", show_lines=True)
        best_table.add_column("Field", style="bold cyan")
        best_table.add_column("Value", style="green")
        best_table.add_row("Model", str(best_row.get("model_name")))
        best_table.add_row("Family", str(best_row.get("model_family")))
        best_table.add_row("Optimize metric", str(best_row.get("optimize_metric")))
        best_table.add_row("Validation wMAPE", str(best_row.get("validation_wmape")))
        best_table.add_row("Validation MAPE", str(best_row.get("validation_mape")))
        best_table.add_row("Training source", str(best_row.get("training_source")))
        best_table.add_row("LLM features used", str(best_row.get("llm_features_used")))
        console.print(best_table)

    per_product = best_registry.loc[best_registry["scope"] == "per-product"].copy()
    if not per_product.empty:
        segment_table = Table(title="Per-Product Winners (Validation)", show_lines=True)
        segment_table.add_column("Segment", style="bold cyan")
        segment_table.add_column("Model", style="green")
        segment_table.add_column("wMAPE", style="green")

        for row in per_product.head(15).itertuples(index=False):
            segment_table.add_row(
                str(getattr(row, "segment_key", "")),
                str(getattr(row, "model_name", "")),
                str(getattr(row, "validation_wmape", "")),
            )
        console.print(segment_table)


@app.command("run-full-pipeline")
def run_full_pipeline_command(
    input_path: Path | None = typer.Option(
        None,
        "--input-path",
        help="Optional explicit source CSV path for data ingestion.",
    ),
    skip_llm: bool = typer.Option(
        False,
        "--skip-llm",
        help="Skip LLM planning/materialization stages.",
    ),
    skip_elasticity: bool = typer.Option(
        False,
        "--skip-elasticity",
        help="Skip causal elasticity fitting stage.",
    ),
    skip_tuning: bool = typer.Option(
        True,
        "--skip-tuning/--run-tuning",
        help="Skip or run Optuna tuning before model training.",
    ),
    segment_level: str = typer.Option(
        "product",
        "--segment-level",
        help="Elasticity segmentation level: product or store-product.",
    ),
    segment_mode: str = typer.Option(
        "global",
        "--segment-mode",
        help="Forecast model training mode: global or per-product.",
    ),
    model: str = typer.Option(
        "all",
        "--model",
        help="Forecast model family: lightgbm, xgboost, or all.",
    ),
    optimize_metric: str = typer.Option(
        "wmape",
        "--optimize-metric",
        help="Optimization metric: wmape, mape, mae, or rmse.",
    ),
    use_llm_features: bool = typer.Option(
        True,
        "--use-llm-features/--no-use-llm-features",
        help="Enable optional LLM feature merging in modeling stages.",
    ),
    random_state: int = typer.Option(
        42,
        "--random-state",
        help="Random seed for tuning/training stages.",
    ),
    n_trials: int = typer.Option(
        20,
        "--n-trials",
        help="Optuna trial count per model when tuning is enabled.",
    ),
    force_refresh_report: bool = typer.Option(
        False,
        "--force-refresh-report",
        help="Force regeneration of final report artifacts.",
    ),
) -> None:
    """Run the full staged pipeline with explicit skip controls and final reporting."""
    try:
        outputs = run_full_pipeline(
            FullPipelineConfig(
                input_path=input_path,
                use_llm_features=use_llm_features,
                skip_llm=skip_llm,
                skip_elasticity=skip_elasticity,
                skip_tuning=skip_tuning,
                segment_level=segment_level,
                optimize_metric=optimize_metric,
                segment_mode=segment_mode,
                model=model,
                random_state=random_state,
                n_trials=n_trials,
                force_refresh_report=force_refresh_report,
            )
        )
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        console.print(f"Full pipeline failed: {exc}", style="bold red")
        raise typer.Exit(code=1) from exc

    result = Table(title="Full Pipeline Artifacts", show_lines=True)
    result.add_column("Artifact", style="bold cyan")
    result.add_column("Path", style="green")
    for key, value in outputs.items():
        result.add_row(key, str(value))
    console.print(result)
    console.print("Full pipeline completed.", style="bold green")


@app.command("forecast-next")
def forecast_next(
    use_llm_features: bool = typer.Option(
        True,
        "--use-llm-features/--no-use-llm-features",
        help="Enable optional LLM feature merging for latest forecast scoring.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output-path",
        help="Optional output CSV path for latest forecasts.",
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input-path",
        help="Optional manual features input override.",
    ),
    llm_features_path: Path | None = typer.Option(
        None,
        "--llm-features-path",
        help="Optional LLM feature parquet override.",
    ),
) -> None:
    """Generate latest forecasts using the current best registered model artifact."""
    try:
        outputs = forecast_next_pipeline(
            ForecastNextConfig(
                use_llm_features=use_llm_features,
                output_path=output_path,
                input_path=input_path,
                llm_features_path=llm_features_path,
            )
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        console.print(f"Forecast generation failed: {exc}", style="bold red")
        raise typer.Exit(code=1) from exc

    result = Table(title="Forecast Generation Artifacts", show_lines=True)
    result.add_column("Artifact", style="bold cyan")
    result.add_column("Path", style="green")
    for key, value in outputs.items():
        result.add_row(key, str(value))
    console.print(result)
    console.print("Latest forecast generation completed.", style="bold green")


@app.command("export-forecasts")
def export_forecasts(
    input_path: Path | None = typer.Option(
        None,
        "--input-path",
        help="Optional source forecast CSV path (defaults to artifacts/forecasts_latest.csv).",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output-path",
        help="Optional export forecast CSV destination path.",
    ),
) -> None:
    """Export forecast predictions to a downstream CSV path."""
    try:
        outputs = export_forecasts_pipeline(
            ForecastExportConfig(
                input_path=input_path,
                output_path=output_path,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Forecast export failed: {exc}", style="bold red")
        raise typer.Exit(code=1) from exc

    result = Table(title="Forecast Export Artifacts", show_lines=True)
    result.add_column("Artifact", style="bold cyan")
    result.add_column("Path", style="green")
    for key, value in outputs.items():
        result.add_row(key, str(value))
    console.print(result)
    console.print("Forecast export completed.", style="bold green")


@app.command("generate-report")
def generate_report(
    optimize_metric: str = typer.Option(
        "wmape",
        "--optimize-metric",
        help="Metric used for best-model reconciliation in final report generation.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output-path",
        help="Optional final markdown report output path.",
    ),
    force_refresh_report: bool = typer.Option(
        False,
        "--force-refresh-report",
        help="Force report regeneration from existing artifacts.",
    ),
) -> None:
    """Regenerate consolidated summary artifacts and final run report."""
    try:
        outputs = generate_reporting_artifacts(
            ReportingConfig(
                optimize_metric=optimize_metric,
                report_output_path=output_path,
                force_refresh_report=force_refresh_report,
            )
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        console.print(f"Report generation failed: {exc}", style="bold red")
        raise typer.Exit(code=1) from exc

    result = Table(title="Final Reporting Artifacts", show_lines=True)
    result.add_column("Artifact", style="bold cyan")
    result.add_column("Path", style="green")
    for key, value in outputs.items():
        result.add_row(key, str(value))
    console.print(result)
    console.print("Final reporting artifacts generated.", style="bold green")


@app.command("run-acceptance-pass")
def run_acceptance_pass(
    enable_llm: bool = typer.Option(
        False,
        "--enable-llm",
        help="Attempt LLM planning/materialization during acceptance pass.",
    ),
    skip_forecast_training: bool = typer.Option(
        False,
        "--skip-forecast-training",
        help="Skip forecast model training and evaluate with existing artifacts/baselines.",
    ),
    skip_elasticity_refit: bool = typer.Option(
        False,
        "--skip-elasticity-refit",
        help="Skip elasticity refit and reuse existing elasticity artifacts.",
    ),
    force_report_refresh: bool = typer.Option(
        False,
        "--force-report-refresh",
        help="Force report and acceptance summary regeneration.",
    ),
) -> None:
    """Run a focused acceptance flow and regenerate final reporting artifacts."""
    outputs: dict[str, Path] = {}

    if enable_llm:
        try:
            outputs.update(plan_llm_features_pipeline(include_manual_input=True))
            outputs.update(build_llm_features_pipeline(include_manual_input=True))
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"LLM acceptance planning/build failed: {exc}", style="bold red")
            raise typer.Exit(code=1) from exc

    if not skip_elasticity_refit:
        try:
            outputs.update(
                fit_elasticity_pipeline(
                    ElasticityRunConfig(use_llm_features=enable_llm)
                )
            )
        except (FileNotFoundError, ValueError, ImportError) as exc:
            console.print(f"Elasticity refit failed: {exc}", style="bold red")
            raise typer.Exit(code=1) from exc

    if not skip_forecast_training:
        try:
            outputs.update(
                train_forecast_models_pipeline(
                    ForecastModelTrainingConfig(
                        use_llm_features=enable_llm,
                        use_tuned_params=False,
                    )
                )
            )
        except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
            console.print(f"Forecast training failed: {exc}", style="bold red")
            raise typer.Exit(code=1) from exc

    try:
        outputs.update(
            evaluate_forecast_models_pipeline(
                ForecastEvaluationConfig(use_llm_features=enable_llm)
            )
        )
        outputs.update(
            generate_reporting_artifacts(
                ReportingConfig(
                    optimize_metric="wmape",
                    force_refresh_report=force_report_refresh,
                    run_config_values={
                        "llm_requested": bool(enable_llm),
                        "use_llm_features": bool(enable_llm),
                        "skip_forecast_training": bool(skip_forecast_training),
                        "skip_elasticity_refit": bool(skip_elasticity_refit),
                    },
                )
            )
        )
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        console.print(f"Acceptance pass failed: {exc}", style="bold red")
        raise typer.Exit(code=1) from exc

    result = Table(title="Acceptance Pass Artifacts", show_lines=True)
    result.add_column("Artifact", style="bold cyan")
    result.add_column("Path", style="green")
    for key, value in outputs.items():
        result.add_row(key, str(value))
    console.print(result)

    acceptance_path = build_project_paths().artifacts_dir / "acceptance_summary.json"
    if acceptance_path.exists():
        acceptance = load_json(acceptance_path)
        acceptance_table = Table(title="Acceptance Snapshot", show_lines=True)
        acceptance_table.add_column("Field", style="bold cyan")
        acceptance_table.add_column("Value", style="green")
        acceptance_table.add_row(
            "Winner",
            str(_as_dict(acceptance.get("forecasting_winner")).get("model_name", "unknown")),
        )
        acceptance_table.add_row(
            "Validation wMAPE",
            str(_as_dict(acceptance.get("validation_metrics")).get("wmape", "nan")),
        )
        acceptance_table.add_row(
            "Test wMAPE",
            str(_as_dict(acceptance.get("test_metrics")).get("wmape", "nan")),
        )
        acceptance_table.add_row(
            "Elasticity warnings present",
            str(acceptance.get("elasticity_warning_presence", False)),
        )
        acceptance_table.add_row("LLM requested", str(acceptance.get("llm_requested", False)))
        acceptance_table.add_row("Ollama reachable", str(acceptance.get("ollama_reachable")))
        acceptance_table.add_row(
            "Planner model available",
            str(acceptance.get("planner_model_available")),
        )
        acceptance_table.add_row(
            "LLM output feature count",
            str(acceptance.get("llm_output_feature_count", 0)),
        )
        acceptance_table.add_row(
            "LLM features actually used",
            str(acceptance.get("llm_features_actually_used", False)),
        )
        console.print(acceptance_table)

    console.print("Acceptance pass completed.", style="bold green")


@app.command("validate-acceptance")
def validate_acceptance() -> None:
    """Validate acceptance artifacts for metadata consistency and print concise truth summary."""
    paths = build_project_paths()
    acceptance_path = paths.artifacts_dir / "acceptance_summary.json"
    final_summary_path = paths.artifacts_dir / "final_project_summary.json"
    manifest_path = paths.artifacts_dir / "run_manifest.json"

    missing = [
        path for path in [acceptance_path, final_summary_path, manifest_path] if not path.exists()
    ]
    if missing:
        for path in missing:
            console.print(f"Missing acceptance artifact: {path}", style="bold red")
        raise typer.Exit(code=1)

    acceptance = load_json(acceptance_path)
    final_summary = load_json(final_summary_path)
    manifest = load_json(manifest_path)

    issues: list[str] = []

    acceptance_winner = str(_as_dict(acceptance.get("forecasting_winner")).get("model_name", ""))
    if acceptance_winner and acceptance_winner != str(final_summary.get("best_model_name", "")):
        issues.append("Best model mismatch between acceptance_summary and final_project_summary")

    acceptance_llm_used = bool(acceptance.get("llm_features_actually_used", False))
    manifest_llm_used = bool(manifest.get("llm_features_actually_used", False))
    if acceptance_llm_used != manifest_llm_used:
        issues.append("LLM usage mismatch between acceptance_summary and run_manifest")

    acceptance_elasticity_warn = bool(acceptance.get("elasticity_warning_presence", False))
    manifest_elasticity_warn = bool(manifest.get("elasticity_inference_warnings_present", False))
    if acceptance_elasticity_warn != manifest_elasticity_warn:
        issues.append("Elasticity warning mismatch between acceptance_summary and run_manifest")

    summary_table = Table(title="Acceptance Validation", show_lines=True)
    summary_table.add_column("Field", style="bold cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Winner", acceptance_winner or "unknown")
    summary_table.add_row(
        "Validation wMAPE",
        str(_as_dict(acceptance.get("validation_metrics")).get("wmape", "nan")),
    )
    summary_table.add_row(
        "Test wMAPE",
        str(_as_dict(acceptance.get("test_metrics")).get("wmape", "nan")),
    )
    summary_table.add_row("LLM requested", str(acceptance.get("llm_requested", False)))
    summary_table.add_row("Ollama reachable", str(acceptance.get("ollama_reachable")))
    summary_table.add_row("Planner model available", str(acceptance.get("planner_model_available")))
    summary_table.add_row(
        "LLM output feature count",
        str(acceptance.get("llm_output_feature_count", 0)),
    )
    summary_table.add_row("LLM features actually used", str(acceptance_llm_used))
    summary_table.add_row("Elasticity warning presence", str(acceptance_elasticity_warn))
    console.print(summary_table)

    if issues:
        issues_table = Table(title="Acceptance Consistency Issues", show_lines=True)
        issues_table.add_column("Issue", style="bold red")
        for issue in issues:
            issues_table.add_row(issue)
        console.print(issues_table)
        raise typer.Exit(code=1)

    console.print("Acceptance artifacts are consistent.", style="bold green")


def main(argv: Sequence[str] | None = None) -> int:
    """Programmatic entrypoint for console scripts and tests.

    Args:
        argv: Optional command arguments.

    Returns:
        Process exit code.
    """
    try:
        app(args=list(argv) if argv is not None else None, standalone_mode=False)
    except typer.Exit as exc:
        return int(exc.exit_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
