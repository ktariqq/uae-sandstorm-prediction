"""
Utility functions — config loading, logging setup, data splitting.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with timestamp and level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    target_col: str = "sandstorm_event",
    feature_cols: list[str] | None = None,
) -> tuple:
    """
    Time-aware train/test split.

    Returns
    -------
    tuple: X_train, X_test, y_train, y_test
    """
    n = len(df)
    split_idx = int(n * (1 - test_size))

    if feature_cols is None:
        exclude = {"timestamp", target_col}
        feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].values
    y = df[target_col].values

    return (
        X[:split_idx], X[split_idx:],
        y[:split_idx], y[split_idx:],
    )


def compute_class_weight(y: np.ndarray) -> float:
    """Compute scale_pos_weight for XGBoost from binary label array."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return n_neg / max(n_pos, 1)


def risk_level_from_score(score: float, low_max: float = 0.30, medium_max: float = 0.60) -> str:
    if score < low_max:
        return "LOW"
    elif score < medium_max:
        return "MEDIUM"
    return "HIGH"