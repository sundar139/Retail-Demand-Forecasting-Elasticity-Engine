# Retail Demand Forecasting Elasticity Engine

## Executive Overview
This repository is a local-first, Windows-friendly retail analytics engine that covers the full lifecycle from data ingestion to forecasting, elasticity analysis, orchestration, prediction export, and final reporting. It is organized in phased, production-oriented modules with strict chronological handling, leakage-safe features, explicit metadata, and reproducible artifacts.

## Problem Statement
Retail demand systems need reliable forecasting and pricing intelligence, but practical implementations often fail due to:
- data leakage from random splits or future-aware features
- fragile pipelines that are difficult to reproduce locally
- inconsistent reporting where narrative and metrics drift apart
- optional AI components that can silently produce misleading metadata

This project addresses those issues with deterministic pipelines, auditable feature policies, strict time-aware evaluation, and source-of-truth reporting.

## Key Capabilities
- Deterministic ingestion, schema validation, and chronological train/validation/test splitting
- Leakage-safe manual feature engineering with split-aware feature artifacts
- Safe, structured LLM feature planning/materialization with deterministic transformation guards
- Segmented causal own-price elasticity estimation using EconML LinearDML
- Baseline-first forecasting benchmarks (naive, seasonal naive, rolling means, median)
- LightGBM and XGBoost model training with optional Optuna tuning
- wMAPE-first model selection and per-product winner registry
- Full orchestration command for end-to-end stage execution
- Best-model prediction workflow and export-ready forecast outputs
- Final manifest and report generation with consistency reconciliation

## Architecture
The package under src/retail_forecasting is modular by phase and responsibility:
- data_validation, data_loading, preprocessing
- features_manual, features_llm, feature_spec, features_common
- causal_utils, causal_dml
- baselines, forecasting_models, evaluation, tuning
- pipeline (Phase 7 orchestration)
- predict (best-model scoring and export)
- reporting (summary artifacts, manifest, final report)
- cli (Typer command surface)

Design principles:
- local-first execution
- explicit artifact contracts between stages
- no random KFold for forecasting evaluation
- no silent fallback behavior
- metadata values derived from actual artifacts and feature matrices

## Dataset
Primary expected source is the Kaggle retail inventory/forecasting-style CSV placed in data/raw.

Canonical required columns:
- date
- store_id
- product_id
- units_sold
- price

Optional columns are supported when available (for example discount, inventory_level, demand_forecast, promotion, holiday, competitor fields).

## Methodology
### Data Ingestion
- validate raw schema and normalize canonical columns
- remove duplicates deterministically
- split chronologically into train/validation/test
- persist split metadata in artifacts/split_summary.json

### Manual Features
- create lag and shifted-rolling demand/price features grouped by store_id and product_id
- enforce leakage safety by using only prior observations for historical statistics
- write:
  - data/processed/features_manual.parquet
  - data/processed/features_train.parquet
  - data/processed/features_validation.parquet
  - data/processed/features_test.parquet

### Safe LLM Feature Planning
- planner only proposes JSON specs; no model-generated executable code is run
- strict schema validation and deterministic pandas-only materialization
- LLM-derived feature columns are optional augmentation only
- if zero validated LLM outputs exist, no LLM columns are merged into modeling matrices

### Causal Elasticity
- segmented own-price elasticity with EconML LinearDML
- log-log setup with safe epsilon transform
- dedicated causal feature profiles (`lean` and `full`) kept separate from forecasting feature richness
- default `lean` profile intentionally excludes own-price lag/rolling/momentum controls, LLM-derived features, and `demand_forecast` unless strict pre-treatment guarantees are explicitly enabled
- stronger guardrails for sample size, non-null treatment/outcome pairs, unique price support, and log-scale variation
- segment quality status persisted (ok, warning_inference_unstable, skipped)

Why causal controls are intentionally leaner than forecasting controls:
- forecasting favors predictive lift, so larger feature sets are often acceptable
- elasticity needs stable identification and valid inference, so compact, auditable, pre-treatment controls are preferred
- reducing collinearity and endogenous proxies lowers underdetermined covariance and invalid inference warnings

### Forecasting Models
- baseline-first comparison:
  - naive_last
  - seasonal_naive_7
  - rolling_mean_7
  - rolling_mean_28
  - median_expanding
- ML models:
  - LightGBM regressor
  - XGBoost regressor
- strict chronological validation/test scoring
- primary selection metric: wMAPE
- optional Optuna tuning with bounded search spaces

## Results Summary
Latest artifact-driven summary (from forecast_metrics_validation.csv and forecast_metrics_test.csv):
- Best overall model: LightGBM
- Validation wMAPE: approximately 5.48
- Test wMAPE: approximately 5.44
- Baselines are substantially weaker than the boosted tree models

Important note on LLM feature metadata:
- LLM features are marked as used only when validated LLM output columns exist and are actually included in the model matrix.
- Passing a CLI flag alone does not set LLM usage to true.

## Setup (Windows + uv)
1. Install Python 3.13 and sync dependencies:
   uv python install 3.13
   uv sync --all-groups --python 3.13

2. Run quality gates:
   uv run --python 3.13 ruff check .
   uv run --python 3.13 mypy src
   uv run --python 3.13 pytest -q

## Ollama Integration
Environment keys in .env or shell:
- OLLAMA_HOST
- OLLAMA_MODEL

When Ollama is unavailable, LLM planning/materialization stages degrade gracefully and record explicit metadata without crashing the entire pipeline.

## CLI Workflows
Core staged commands:
- uv run --python 3.13 retail-forecasting-engine validate-data
- uv run --python 3.13 retail-forecasting-engine prepare-data
- uv run --python 3.13 retail-forecasting-engine build-manual-features
- uv run --python 3.13 retail-forecasting-engine plan-llm-features
- uv run --python 3.13 retail-forecasting-engine build-llm-features
- uv run --python 3.13 retail-forecasting-engine fit-elasticity
- uv run --python 3.13 retail-forecasting-engine run-baselines
- uv run --python 3.13 retail-forecasting-engine tune-forecast-models
- uv run --python 3.13 retail-forecasting-engine train-forecast-models
- uv run --python 3.13 retail-forecasting-engine evaluate-forecast-models
- uv run --python 3.13 retail-forecasting-engine summarize-forecasting

Phase 7 orchestration and productionization commands:
- uv run --python 3.13 retail-forecasting-engine run-full-pipeline
- uv run --python 3.13 retail-forecasting-engine forecast-next
- uv run --python 3.13 retail-forecasting-engine export-forecasts
- uv run --python 3.13 retail-forecasting-engine generate-report

Useful options:
- --skip-llm
- --skip-elasticity
- --skip-tuning
- --segment-level
- --feature-profile [lean|full]
- --segment-mode
- --use-llm-features / --no-use-llm-features
- --optimize-metric
- --n-trials
- --output-path
- --force-refresh-report

## Project Structure
- src/retail_forecasting/: core package
- tests/: automated tests (synthetic/lightweight fixtures)
- data/: raw/interim/processed datasets
- artifacts/: machine-readable outputs and registries
- reports/: visual and markdown summaries
- notebooks/: concise demonstration notebooks that call package APIs

## Artifacts and Reports
Primary outputs include:
- artifacts/forecast_metrics_validation.csv
- artifacts/forecast_metrics_test.csv
- artifacts/forecast_segment_metrics.csv
- artifacts/forecast_metrics_summary.csv
- artifacts/best_model_registry.csv
- artifacts/model_training_registry.csv
- artifacts/optuna_study_summary.csv
- artifacts/elasticity_estimates.csv
- artifacts/elasticity_summary.csv
- artifacts/run_manifest.json
- artifacts/final_project_summary.json
- artifacts/forecasts_latest.csv
- artifacts/forecast_export_summary.json
- reports/final_run_report.md

## Limitations and Caveats
- Data is synthetic-like and should not be interpreted as calibrated production demand behavior.
- Observational pricing data limits strict causal interpretation, even with DML controls.
- Forecasting quality can drift with assortment/promo regime changes and requires monitoring/retraining.
- Optional LLM planning does not guarantee useful feature proposals.

## Future Work
- Add scheduled retraining and model drift monitoring with threshold alerts
- Add richer backtesting windows and probabilistic forecast intervals
- Add configurable scenario simulation for pricing and promotion plans
- Add packaging/deployment templates for local services and batch jobs
- Add data contracts and lineage tracking for stronger production governance
