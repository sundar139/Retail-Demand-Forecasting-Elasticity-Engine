# Final Run Report

Generated: 2026-04-09T00:12:29+00:00

## Data Preparation
- Source file: retail_store_inventory.csv
- Rows: 73100
- Date range: 2022-01-01 to 2024-01-01
- Validation start: 2023-05-27
- Test start: 2023-09-14

## Feature Generation
- Manual feature count: 34
- LLM accepted specs: 2
- LLM requested: True
- Ollama reachable: True
- Planner model available: True
- LLM feature file exists: True
- LLM output feature count: 2
- LLM features actually used downstream: True
- LLM feature columns used: price_lag_14, price_roll_mean_14

## Elasticity Summary
- Segments attempted: 20
- Successful fits: 20
- Skipped fits: 0
- Failed fits: 0
- Inference warnings present: True
- Inference warning count: 20
- CI caution present: True
- Quality counts:
  - warning_inference_unstable: 20

## Forecasting Model Comparison
- lightgbm: validation wMAPE=5.359179, test wMAPE=5.311055
- xgboost: validation wMAPE=5.383167, test wMAPE=5.329801
- median_expanding: validation wMAPE=63.750974, test wMAPE=63.020858
- rolling_mean_28: validation wMAPE=66.483794, test wMAPE=65.616509
- rolling_mean_7: validation wMAPE=69.134261, test wMAPE=68.140793
- seasonal_naive_7: validation wMAPE=88.103376, test wMAPE=86.714948
- naive_last: validation wMAPE=89.074125, test wMAPE=87.471215

## Best Model
- Winner: lightgbm
- Validation wMAPE: 5.359178546375507
- Test wMAPE: 5.311055009769537

## Caveats
- Dataset remains synthetic and should not be treated as calibrated production demand truth.
- Zero-demand days can distort percentage metrics; wMAPE remains primary optimization metric.
- Observational pricing limits causal interpretation confidence even with DML controls.

## Artifact Locations
- forecast_metrics_summary_csv: C:\Users\rohit\Documents\Personal Projects\Retail Demand Forecasting Elasticity Engine\artifacts\forecast_metrics_summary.csv
- best_model_registry_csv: C:\Users\rohit\Documents\Personal Projects\Retail Demand Forecasting Elasticity Engine\artifacts\best_model_registry.csv
- elasticity_summary_csv: C:\Users\rohit\Documents\Personal Projects\Retail Demand Forecasting Elasticity Engine\artifacts\elasticity_summary.csv
- final_project_summary_json: C:\Users\rohit\Documents\Personal Projects\Retail Demand Forecasting Elasticity Engine\artifacts\final_project_summary.json
- run_manifest_json: C:\Users\rohit\Documents\Personal Projects\Retail Demand Forecasting Elasticity Engine\artifacts\run_manifest.json
- acceptance_summary_json: C:\Users\rohit\Documents\Personal Projects\Retail Demand Forecasting Elasticity Engine\artifacts\acceptance_summary.json
