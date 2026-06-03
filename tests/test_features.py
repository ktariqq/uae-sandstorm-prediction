"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.engineering import engineer_features, get_feature_columns
from src.features.labeling import label_sandstorm_events


def make_dummy_df(n=200):
    timestamps = pd.date_range("2022-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "timestamp": timestamps,
        "wind_u10m": rng.normal(3, 2, n),
        "wind_v10m": rng.normal(1.5, 1.5, n),
        "temperature_2m": rng.normal(30, 5, n),
        "humidity": rng.uniform(20, 80, n),
        "surface_pressure": rng.normal(101325, 300, n),
        "aerosol_optical_depth": rng.exponential(0.2, n),
        "visibility_m": rng.uniform(500, 9000, n),
    })


def test_engineer_features_runs():
    df = make_dummy_df()
    out = engineer_features(df, lag_hours=[1, 3], rolling_windows=[3, 6])
    assert "wind_speed" in out.columns
    assert "wind_direction" in out.columns
    assert len(out) > 0


def test_lag_features_created():
    df = make_dummy_df()
    out = engineer_features(df, lag_hours=[1, 3], rolling_windows=[3])
    assert "wind_speed_lag_1h" in out.columns
    assert "wind_speed_lag_3h" in out.columns


def test_rolling_features_created():
    df = make_dummy_df()
    out = engineer_features(df, lag_hours=[1], rolling_windows=[3, 6, 12])
    assert "wind_speed_roll_3h" in out.columns
    assert "wind_speed_roll_12h" in out.columns


def test_cyclical_features():
    df = make_dummy_df()
    out = engineer_features(df, cyclical=True)
    for col in ["month_sin", "month_cos", "doy_sin", "doy_cos"]:
        assert col in out.columns
        assert out[col].between(-1, 1).all()


def test_labeling_both_conditions():
    df = make_dummy_df()
    df_feat = engineer_features(df)
    labeled = label_sandstorm_events(df_feat, visibility_threshold=1000, wind_speed_threshold=8.0)
    assert "sandstorm_event" in labeled.columns
    assert labeled["sandstorm_event"].isin([0, 1]).all()


def test_labeling_positive_rate_nonzero():
    df = make_dummy_df(500)
    # Force some sandstorm conditions
    df.loc[:50, "visibility_m"] = 200.0
    df_feat = engineer_features(df)
    labeled = label_sandstorm_events(df_feat, visibility_threshold=500, wind_speed_threshold=2.0, require_both=False)
    assert labeled["sandstorm_event"].sum() > 0


def test_get_feature_columns_excludes_target():
    df = make_dummy_df()
    df_feat = engineer_features(df)
    labeled = label_sandstorm_events(df_feat)
    cols = get_feature_columns(labeled)
    assert "sandstorm_event" not in cols
    assert "timestamp" not in cols