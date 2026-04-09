# Retail Demand Forecasting Elasticity Engine

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Runtime](https://img.shields.io/badge/Runtime-Local%20First-2d7d46)
![Forecasting](https://img.shields.io/badge/Forecasting-LightGBM%20%7C%20XGBoost-0a6e8a)
![Causal](https://img.shields.io/badge/Causal-EconML%20LinearDML-8a5a00)
![LLM](https://img.shields.io/badge/Ollama-qwen2.5%3A7b-444)

Local-first retail analytics pipeline for:
- chronological demand forecasting
- segmented own-price elasticity estimation
- optional LLM feature planning with deterministic, leakage-safe materialization
- artifact-first reporting and acceptance validation

## Executive Overview
This project implements an end-to-end retail workflow from raw CSV ingestion to model evaluation, elasticity analysis, and export artifacts.

Core design goals:
- strict time-aware evaluation (no random split leakage)
- deterministic feature transformations
- explicit stage artifacts that can be audited after every run
- optional LLM integration that cannot execute generated code

## Architecture and Workflow
```mermaid
flowchart TD
    A[data/raw/retail_store_inventory.csv] --> B[prepare-data]
    B --> C[build-manual-features]
    C --> D[plan-llm-features (optional)]
    D --> E[build-llm-features (optional)]
    C --> F[fit-elasticity]
    E --> F
    C --> G[run-baselines]
    E --> H[train-forecast-models]
    H --> I[evaluate-forecast-models]
    G --> I
    F --> J[generate-report / acceptance artifacts]
    I --> J
    J --> K[forecast-next]
    K --> L[export-forecasts]
```

Main package modules are under `src/retail_forecasting`:
- data and schema: `data_loading.py`, `data_validation.py`, `preprocessing.py`, `schemas.py`
- feature generation: `features_manual.py`, `features_llm.py`, `feature_spec.py`
- causal: `causal_utils.py`, `causal_dml.py`
- forecasting: `baselines.py`, `forecasting_models.py`, `evaluation.py`, `tuning.py`
- orchestration and reporting: `pipeline.py`, `reporting.py`, `predict.py`, `cli.py`

## Tech Stack
| Area | Implementation |
| --- | --- |
| Language/runtime | Python 3.13 (`>=3.13,<3.14`) |
| Data processing | pandas, NumPy, pyarrow |
| Forecasting models | LightGBM, XGBoost, scikit-learn |
| Causal inference | EconML LinearDML |
| Hyperparameter tuning | Optuna |
| CLI and UX | Typer + Rich |
| Quality gates | Ruff, mypy, pytest |
| LLM planner backend | Ollama |

Current Ollama configuration (from `.env.example` and `.env`):
- `OLLAMA_HOST=http://127.0.0.1:11434`
- `OLLAMA_MODEL=qwen2.5:7b`

## Repository Structure
```text
.
|-- src/retail_forecasting/
|-- tests/
|-- data/
|   |-- raw/retail_store_inventory.csv
|   |-- interim/cleaned_retail.parquet
|   `-- processed/*.parquet
|-- artifacts/
|-- reports/
|   |-- final_run_report.md
|   `-- figures/
|-- notebooks/
|-- prompts/
`-- pyproject.toml
```

## Setup
### 1) Environment and dependencies
```bash
uv python install 3.13
uv sync --all-groups --python 3.13
```

### 2) Configure Ollama (optional, but required for LLM feature planning)
```bash
# .env
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b
```

### 3) Quality checks
```bash
uv run --python 3.13 ruff check .
uv run --python 3.13 mypy src
uv run --python 3.13 pytest -q
```

## Command Reference (`python -m retail_forecasting.cli`)
Run from repository root inside the synced environment.

### Stage-by-stage workflow
```bash
python -m retail_forecasting.cli validate-data
python -m retail_forecasting.cli prepare-data
python -m retail_forecasting.cli build-manual-features
python -m retail_forecasting.cli plan-llm-features
python -m retail_forecasting.cli build-llm-features
python -m retail_forecasting.cli fit-elasticity --feature-profile lean
python -m retail_forecasting.cli run-baselines
python -m retail_forecasting.cli train-forecast-models
python -m retail_forecasting.cli evaluate-forecast-models
python -m retail_forecasting.cli generate-report
```

### One-shot orchestration
```bash
python -m retail_forecasting.cli run-full-pipeline
```

### Acceptance refresh and consistency check
```bash
python -m retail_forecasting.cli run-acceptance-pass --enable-llm --force-report-refresh
python -m retail_forecasting.cli validate-acceptance
```

### Forecast scoring and export
```bash
python -m retail_forecasting.cli forecast-next
python -m retail_forecasting.cli export-forecasts
```

## Verified Results (Artifact-Sourced)
All values below come from tracked artifacts in `artifacts/`.

### Forecasting leaderboard (overall)
Source: `artifacts/forecast_metrics_summary.csv`, `artifacts/best_model_registry.csv`

| Model | Validation wMAPE | Test wMAPE | Validation MAE | Test MAE |
| --- | ---: | ---: | ---: | ---: |
| lightgbm (winner) | 5.359178546375507 | 5.311055009769537 | 7.326425807179027 | 7.246171700201857 |
| xgboost | 5.383167136852199 | 5.329800903683682 | 7.359220129447904 | 7.271747779855674 |
| best baseline (median_expanding) | 63.75097420647059 | 63.02085832014029 | 87.15268181818182 | 85.98290909090909 |

Winner details:
- model: `lightgbm`
- optimize metric: `wmape`
- `llm_features_used=True`
- LLM columns in winner: `price_lag_14`, `price_roll_mean_14`

### LLM feature planning and acceptance
Source: `artifacts/llm_features_summary.json`, `artifacts/acceptance_summary.json`

- planner host/model: `http://127.0.0.1:11434` / `qwen2.5:7b`
- raw specs proposed: `4`
- accepted specs: `2`
- rejected specs: `2`
- materialized output features: `2`
- output features: `price_lag_14`, `price_roll_mean_14`
- `ollama_reachable=True`, `planner_model_available=True`
- acceptance summary reports `llm_features_actually_used=True`

### Causal elasticity (latest lean-profile run)
Source: `artifacts/elasticity_run_summary.json`, `artifacts/elasticity_estimates.csv`

- segmentation level: `product`
- feature profile: `lean`
- segments attempted: `20`
- successful fits: `20`
- skipped fits: `0`
- inference warnings present: `False` (`inference_warning_count=0`)
- controls used per segment: `7`
- quality status counts: `ok=20`

Control set used in lean profile:
- `day_of_week`, `month`, `is_weekend`
- `units_sold_lag_7`, `units_sold_lag_28`
- `discount`, `inventory_level`

## Artifact Guide
| Artifact | Purpose |
| --- | --- |
| `artifacts/split_summary.json` | Chronological split boundaries and row counts |
| `artifacts/features_manual_summary.json` | Manual feature generation log and leakage notes |
| `artifacts/llm_feature_plan_raw.json` | Raw planner response and prompt metadata |
| `artifacts/llm_feature_plan_validated.json` | Validated/rejected LLM specs |
| `artifacts/llm_features_summary.json` | LLM acceptance and materialization summary |
| `artifacts/elasticity_estimates.csv` | Segment-level DML estimates and CI bounds |
| `artifacts/elasticity_run_summary.json` | Elasticity run-level diagnostics and guardrail status |
| `artifacts/forecast_metrics_summary.csv` | Unified forecasting leaderboard |
| `artifacts/best_model_registry.csv` | Selected winners (overall and per-product) |
| `artifacts/run_manifest.json` | Consolidated run metadata and selected configuration |
| `artifacts/acceptance_summary.json` | Acceptance snapshot for winner/LLM/elasticity summary flags |
| `artifacts/final_project_summary.json` | Final high-level report summary |
| `reports/final_run_report.md` | Human-readable report output |

## Caveats and Limitations
- Dataset realism: current data is a retail benchmark-style dataset and should not be treated as calibrated production demand behavior.
- Causal interpretation: elasticity is estimated from observational data, so coefficients are conditional estimates rather than guaranteed structural effects.
- Artifact freshness: some summary artifacts can reflect an earlier run if stages are executed independently.
  - Example in current workspace: `elasticity_run_summary.json` shows the latest lean-profile run with zero inference warnings, while `elasticity_summary.csv` and `final_project_summary.json` still contain an earlier warning-heavy snapshot.
  - Recommended fix: rerun `run-acceptance-pass --force-report-refresh` after major stage reruns.
- LLM optionality: LLM planning can be disabled, skipped, or rejected by schema validation; forecasting and elasticity still run with manual features.

## Conclusion
This repository demonstrates a full local analytics workflow with auditable artifacts, strict temporal evaluation, optional but constrained LLM augmentation, and separate forecasting vs causal-control design choices. The current artifact set shows a strong forecasting winner (LightGBM), accepted/used LLM feature additions, and a stable latest lean-profile elasticity refit.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
