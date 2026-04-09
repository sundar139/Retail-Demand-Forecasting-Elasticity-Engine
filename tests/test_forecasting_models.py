"""Tests for forecasting data loading and model wrappers."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from retail_forecasting.forecasting_models import (
    MODEL_LIGHTGBM,
    MODEL_XGBOOST,
    build_forecast_model,
    load_forecasting_data_bundle,
    prepare_forecasting_matrices,
)
from retail_forecasting.paths import ProjectPaths


def _build_temp_paths(tmp_path: Path) -> ProjectPaths:
    data_raw = tmp_path / "data" / "raw"
    data_interim = tmp_path / "data" / "interim"
    data_processed = tmp_path / "data" / "processed"
    artifacts = tmp_path / "artifacts"
    notebooks = tmp_path / "notebooks"
    reports = tmp_path / "reports" / "figures"
    prompts = tmp_path / "prompts"
    scripts = tmp_path / "scripts"

    for directory in [
        data_raw,
        data_interim,
        data_processed,
        artifacts,
        notebooks,
        reports,
        prompts,
        scripts,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        project_root=tmp_path,
        data_raw_dir=data_raw,
        data_interim_dir=data_interim,
        data_processed_dir=data_processed,
        artifacts_dir=artifacts,
        notebooks_dir=notebooks,
        reports_figures_dir=reports,
        prompts_dir=prompts,
        scripts_dir=scripts,
    )


def _write_feature_split_artifacts(paths: ProjectPaths) -> None:
    dates = pd.date_range("2024-01-01", periods=64, freq="D")
    rows: list[dict[str, object]] = []

    for store_id in ["S1", "S2"]:
        for product_id in ["P1", "P2"]:
            for day_idx, date_value in enumerate(dates, start=1):
                units = float(20 + day_idx + (3 if product_id == "P2" else 0))
                rows.append(
                    {
                        "date": date_value,
                        "store_id": store_id,
                        "product_id": product_id,
                        "units_sold": units,
                        "price": float(8.0 + (day_idx % 7) * 0.15),
                        "discount": 0.1 if day_idx % 5 == 0 else 0.0,
                        "units_sold_lag_1": np.nan if day_idx == 1 else units - 1.0,
                        "units_sold_roll_mean_7": float(max(units - 3.0, 0.0)),
                    }
                )

    full_frame = pd.DataFrame(rows).sort_values(
        ["store_id", "product_id", "date"],
        kind="mergesort",
    )

    validation_start = pd.Timestamp("2024-02-20")
    test_start = pd.Timestamp("2024-03-01")

    train = full_frame.loc[full_frame["date"] < validation_start].reset_index(drop=True)
    validation = full_frame.loc[
        (full_frame["date"] >= validation_start) & (full_frame["date"] < test_start)
    ].reset_index(drop=True)
    test = full_frame.loc[full_frame["date"] >= test_start].reset_index(drop=True)

    full_frame.to_parquet(paths.data_processed_dir / "features_manual.parquet", index=False)
    train.to_parquet(paths.data_processed_dir / "features_train.parquet", index=False)
    validation.to_parquet(paths.data_processed_dir / "features_validation.parquet", index=False)
    test.to_parquet(paths.data_processed_dir / "features_test.parquet", index=False)


def test_load_bundle_handles_missing_llm_features_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bundle loading should remain valid when optional LLM file is absent."""
    from retail_forecasting import forecasting_models

    paths = _build_temp_paths(tmp_path)
    _write_feature_split_artifacts(paths)
    monkeypatch.setattr(forecasting_models, "build_project_paths", lambda: paths)

    bundle = load_forecasting_data_bundle(use_llm_features=True)

    assert bundle.training_source == "split_feature_artifacts"
    assert bundle.llm_added_columns == []
    assert any("file not found" in note for note in bundle.notes)


def test_prepare_matrices_excludes_identifier_and_date_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matrix preparation should exclude raw identifiers/date but include encoded IDs."""
    from retail_forecasting import forecasting_models

    paths = _build_temp_paths(tmp_path)
    _write_feature_split_artifacts(paths)
    monkeypatch.setattr(forecasting_models, "build_project_paths", lambda: paths)

    bundle = load_forecasting_data_bundle(use_llm_features=False)
    matrices = prepare_forecasting_matrices(bundle)

    assert "date" not in matrices.feature_columns
    assert "store_id" not in matrices.feature_columns
    assert "product_id" not in matrices.feature_columns
    assert "store_id_encoded" in matrices.feature_columns
    assert "product_id_encoded" in matrices.feature_columns

    assert matrices.X_train.shape[0] == bundle.train_frame.shape[0]
    assert matrices.X_validation.shape[0] == bundle.validation_frame.shape[0]
    assert matrices.X_test.shape[0] == bundle.test_frame.shape[0]


def test_model_wrappers_fit_predict_shape_consistency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both model wrappers should produce prediction arrays aligned to input rows."""
    from retail_forecasting import forecasting_models

    paths = _build_temp_paths(tmp_path)
    _write_feature_split_artifacts(paths)
    monkeypatch.setattr(forecasting_models, "build_project_paths", lambda: paths)

    bundle = load_forecasting_data_bundle(use_llm_features=False)
    matrices = prepare_forecasting_matrices(bundle)

    for model_name in [MODEL_LIGHTGBM, MODEL_XGBOOST]:
        try:
            model = build_forecast_model(
                model_name,
                random_state=42,
                params={"n_estimators": 20, "max_depth": 4},
            )
        except ImportError:
            pytest.skip(f"Optional dependency for {model_name} is not available")

        model.fit(matrices.X_train, matrices.y_train)
        predictions = model.predict(matrices.X_validation)
        assert predictions.shape == (matrices.X_validation.shape[0],)


def test_llm_used_flag_requires_columns_in_model_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM metadata flag must remain false when merged LLM columns are not usable model features."""
    from retail_forecasting import forecasting_models

    paths = _build_temp_paths(tmp_path)
    _write_feature_split_artifacts(paths)

    # Non-numeric LLM column should merge but be excluded from the model matrix.
    llm_source = pd.read_parquet(paths.data_processed_dir / "features_manual.parquet")
    llm_frame = llm_source[["date", "store_id", "product_id"]].copy(deep=True)
    llm_frame["llm_textual_signal"] = "note"
    llm_frame.to_parquet(paths.data_processed_dir / "features_llm.parquet", index=False)

    monkeypatch.setattr(forecasting_models, "build_project_paths", lambda: paths)

    bundle = load_forecasting_data_bundle(use_llm_features=True)
    assert bundle.llm_added_columns == ["llm_textual_signal"]

    matrices = prepare_forecasting_matrices(bundle)
    assert matrices.llm_feature_columns_in_matrix == []
    assert matrices.llm_features_used is False


def test_llm_summary_zero_output_blocks_stale_llm_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If LLM summary reports zero output features, stale LLM parquet columns must not be merged."""
    from retail_forecasting import forecasting_models

    paths = _build_temp_paths(tmp_path)
    _write_feature_split_artifacts(paths)

    llm_source = pd.read_parquet(paths.data_processed_dir / "features_manual.parquet")
    llm_frame = llm_source[["date", "store_id", "product_id"]].copy(deep=True)
    llm_frame["stale_llm_signal"] = 1.0
    llm_frame.to_parquet(paths.data_processed_dir / "features_llm.parquet", index=False)

    (paths.artifacts_dir / "llm_features_summary.json").write_text(
        json.dumps(
            {
                "output_feature_count": 0,
                "output_feature_names": [],
                "accepted_spec_count": 0,
                "ollama_reachable": True,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(forecasting_models, "build_project_paths", lambda: paths)

    bundle = load_forecasting_data_bundle(use_llm_features=True)
    assert bundle.llm_added_columns == []

    matrices = prepare_forecasting_matrices(bundle)
    assert "stale_llm_signal" not in matrices.feature_columns
    assert matrices.llm_features_used is False
