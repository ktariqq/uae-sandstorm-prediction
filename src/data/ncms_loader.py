"""
NCMS (National Centre of Meteorology, UAE) Data Loader.

Expected CSV columns:
  timestamp, wind_speed_obs, wind_direction_obs, visibility_m

If actual NCMS API access is unavailable, use the sample generator:
  data/sample/generate_sample_data.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLS = ["timestamp", "wind_speed_obs", "wind_direction_obs", "visibility_m"]
OPTIONAL_LABEL_COL = "sandstorm_event"

def load_ncms(path: str | Path) -> pd.DataFrame:
    """
    Load NCMS ground-observation CSV data.

    Parameters
    ----------
    path : str or Path
        Path to NCMS CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned NCMS observations.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NCMS file not found: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = _validate_and_clean(df)

    logger.info(f"NCMS loaded: {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"NCMS missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Physical range enforcement
    if "visibility_m" in df.columns:
        df["visibility_m"] = df["visibility_m"].clip(0, 70000)
    if "wind_speed_obs" in df.columns:
        df["wind_speed_obs"] = df["wind_speed_obs"].clip(0, 80)
    if "wind_direction_obs" in df.columns:
        df["wind_direction_obs"] = df["wind_direction_obs"].clip(0, 360)

    df = df.dropna(subset=REQUIRED_COLS)
# Pass through real labels if present
    if OPTIONAL_LABEL_COL in df.columns:
        df[OPTIONAL_LABEL_COL] = df[OPTIONAL_LABEL_COL].fillna(0).astype(int)
    return df

def merge_datasets(merra2: pd.DataFrame, ncms: pd.DataFrame) -> pd.DataFrame:
    """
    Merge MERRA-2 and NCMS datasets on timestamp using nearest-time join.

    Parameters
    ----------
    merra2 : pd.DataFrame
    ncms : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Merged DataFrame aligned by timestamp.
    """
    merra2 = merra2.set_index("timestamp")
    ncms = ncms.set_index("timestamp")

    # Resample both to hourly and merge
    merra2_hourly = merra2.resample("1h").mean()
    ncms_hourly = ncms.resample("1h").mean()

    merged = merra2_hourly.join(ncms_hourly, how="inner")
    merged = merged.reset_index().rename(columns={"index": "timestamp"})

    logger.info(f"Merged dataset: {len(merged)} aligned hourly records")
    return merged