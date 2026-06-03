"""
Sandstorm event labeling from meteorological thresholds.

If no pre-labeled ground truth is available, events are defined as:
  visibility < visibility_threshold (meters)
  AND wind_speed > wind_speed_threshold (m/s)

This proxy labeling strategy is documented and configurable via config.yaml.
"""

from __future__ import annotations

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def label_sandstorm_events(
    df: pd.DataFrame,
    visibility_threshold: float = 1000.0,
    wind_speed_threshold: float = 8.0,
    require_both: bool = True,
    target_col: str = "sandstorm_event",
) -> pd.DataFrame:
    """
    Assign binary sandstorm event labels based on meteorological thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'visibility_m' and 'wind_speed' columns.
    visibility_threshold : float
        Visibility (meters) below which poor conditions are flagged.
    wind_speed_threshold : float
        Wind speed (m/s) above which strong wind conditions are flagged.
    require_both : bool
        If True, both conditions must hold (AND logic).
        If False, either condition triggers a label (OR logic).
    target_col : str
        Name for the output label column.

    Returns
    -------
    pd.DataFrame
        DataFrame with binary label column appended.
    """
    df = df.copy()

    # -----------------------------
    # NEW: Use real labels if available
    # -----------------------------
    if target_col in df.columns:
        logger.info("Real event labels found — skipping threshold labeling.")
        return df

    has_visibility = "visibility_m" in df.columns
    has_wind = "wind_speed" in df.columns

    if not has_visibility and not has_wind:
        raise ValueError(
            "DataFrame must contain 'visibility_m' and/or 'wind_speed' for labeling."
        )

    low_vis = pd.Series(True, index=df.index)
    high_wind = pd.Series(True, index=df.index)

    if has_visibility:
        low_vis = df["visibility_m"] < visibility_threshold
    else:
        logger.warning("'visibility_m' not found — using wind speed only for labeling.")

    if has_wind:
        high_wind = df["wind_speed"] > wind_speed_threshold
    else:
        logger.warning("'wind_speed' not found — using visibility only for labeling.")

    if require_both and has_visibility and has_wind:
        df[target_col] = (low_vis & high_wind).astype(int)
    else:
        df[target_col] = (low_vis | high_wind).astype(int)

    event_count = df[target_col].sum()
    total = len(df)
    rate = event_count / total * 100

    logger.info(
        f"Labels assigned: {event_count} sandstorm events out of {total} records "
        f"({rate:.2f}% positive rate)"
    )

    return df