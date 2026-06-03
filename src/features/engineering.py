"""
Feature engineering pipeline for sandstorm prediction.

Implements:
  - Wind speed magnitude and direction from u/v components
  - Lag features (configurable)
  - Rolling window averages (configurable)
  - Humidity-pressure interaction
  - Cyclical seasonal encoding
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def engineer_features(
    df: pd.DataFrame,
    lag_hours: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    cyclical: bool = True,
) -> pd.DataFrame:

    if lag_hours is None:
        lag_hours = [1, 3, 6]
    if rolling_windows is None:
        rolling_windows = [3, 6, 12]

    df = df.copy()

    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        ts = df["timestamp"]
    else:
        ts = None

    df = _wind_features(df)
    df = _lag_features(df, lag_hours)
    df = _rolling_features(df, rolling_windows)
    df = _interaction_features(df)

    if cyclical and ts is not None:
        df = _cyclical_features(df, ts)

    before = len(df)
    df = df.dropna()

    logger.info(
        f"Feature engineering complete. Rows before/after NaN drop: {before} → {len(df)}"
    )

    return df


def _wind_features(df: pd.DataFrame) -> pd.DataFrame:
    if "wind_u10m" in df.columns and "wind_v10m" in df.columns:
        df["wind_speed"] = np.sqrt(df["wind_u10m"] ** 2 + df["wind_v10m"] ** 2)
        df["wind_direction"] = (
            np.degrees(np.arctan2(-df["wind_u10m"], -df["wind_v10m"])) % 360
        )
    elif "wind_speed_obs" in df.columns:
        df["wind_speed"] = df["wind_speed_obs"]
        df["wind_direction"] = df.get("wind_direction_obs", np.nan)
    else:
        raise ValueError("Missing wind data columns.")

    return df


def _lag_features(df: pd.DataFrame, lag_hours: list[int]) -> pd.DataFrame:
    lag_cols = ["wind_speed", "humidity", "aerosol_optical_depth"]
    available = [c for c in lag_cols if c in df.columns]

    for col in available:
        for lag in lag_hours:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

    return df


def _rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    if "wind_speed" in df.columns:
        for w in windows:
            df[f"wind_speed_roll_{w}h"] = df["wind_speed"].rolling(w, min_periods=1).mean()

    if "humidity" in df.columns:
        for w in windows:
            df[f"humidity_roll_{w}h"] = df["humidity"].rolling(w, min_periods=1).mean()

    return df


def _interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    if "humidity" in df.columns and "surface_pressure" in df.columns:
        df["humidity_pressure_ratio"] = df["humidity"] / (df["surface_pressure"] / 1000.0)

    if "aerosol_optical_depth" in df.columns and "wind_speed" in df.columns:
        df["aod_wind_interaction"] = df["aerosol_optical_depth"] * df["wind_speed"]

    return df


def _cyclical_features(df: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
    df["month_sin"] = np.sin(2 * np.pi * timestamps.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * timestamps.dt.month / 12)

    df["doy_sin"] = np.sin(2 * np.pi * timestamps.dt.dayofyear / 365)
    df["doy_cos"] = np.cos(2 * np.pi * timestamps.dt.dayofyear / 365)

    df["hour_sin"] = np.sin(2 * np.pi * timestamps.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * timestamps.dt.hour / 24)

    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = "sandstorm_event") -> list[str]:
    exclude = {
        "timestamp",
        target_col,
        "wind_u10m",
        "wind_v10m",
        "wind_speed_obs",
        "wind_direction_obs",
    }
    return [c for c in df.columns if c not in exclude]