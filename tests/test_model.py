"""Tests for model training and inference."""

import numpy as np
import pandas as pd
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.engineering import engineer_features, get_feature_columns
from src.features.labeling import label_sandstorm_events
from src.model.train import train_model
from src.model.inference import SandstormPredictor


def make_feature_df(n=500, seed=42):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2022-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "wind_u10m": rng.normal(3, 2, n),
        "wind_v10m": rng.normal(1.5, 1.5, n),
        "temperature_2m": rng.normal(30, 5, n),
        "humidity": rng.uniform(20, 80, n),
        "surface_pressure": rng.normal(101325, 300, n),
        "aerosol_optical_depth": rng.exponential(0.2, n),
        "visibility_m": rng.uniform(200, 9000, n),
    })
    df = engineer_features(df, lag_hours=[1, 3], rolling_windows=[3, 6])
    df = label_sandstorm_events(df, visibility_threshold=2000, wind_speed_threshold=5.0, require_both=False)
    return df


def test_train_model_returns_metrics():
    df = make_feature_df()
    feature_cols = get_feature_columns(df)
    model, metrics = train_model(df, feature_cols=feature_cols, config={
        "test_size": 0.2,
        "xgboost": {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": 42,
        }
    })
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_train_saves_model():
    df = make_feature_df()
    feature_cols = get_feature_columns(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.pkl"
        model, _ = train_model(df, feature_cols=feature_cols, model_path=model_path, config={
            "test_size": 0.2,
            "xgboost": {
                "n_estimators": 5,
                "max_depth": 2,
                "learning_rate": 0.1,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "random_state": 42,
            }
        })
        assert model_path.exists()

        predictor = SandstormPredictor.from_saved(model_path)
        result = predictor.predict({col: 0.5 for col in feature_cols})
        assert 0.0 <= result.risk_score <= 1.0
        assert result.risk_level in {"LOW", "MEDIUM", "HIGH"}


def test_inference_risk_levels():
    df = make_feature_df()
    feature_cols = get_feature_columns(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.pkl"
        train_model(df, feature_cols=feature_cols, model_path=model_path, config={
            "test_size": 0.2,
            "xgboost": {
                "n_estimators": 5, "max_depth": 2, "learning_rate": 0.1,
                "objective": "binary:logistic", "eval_metric": "auc", "random_state": 42,
            }
        })
        predictor = SandstormPredictor.from_saved(model_path)

        high_result = predictor.predict({col: 99.9 for col in feature_cols})
        assert high_result.risk_level in {"LOW", "MEDIUM", "HIGH"}

        zero_result = predictor.predict({col: 0.0 for col in feature_cols})
        assert zero_result.risk_level in {"LOW", "MEDIUM", "HIGH"}