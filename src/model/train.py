"""
XGBoost training module for sandstorm prediction.

Implements:
  - Time-based train/test split
  - Class imbalance handling via scale_pos_weight
  - Full evaluation suite (accuracy, precision, recall, F1, ROC-AUC)
  - Model persistence via joblib
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "sandstorm_event",
    config: dict | None = None,
    model_path: str | Path | None = None,
) -> tuple[XGBClassifier, dict[str, Any]]:
    """
    Train an XGBoost binary classifier for sandstorm prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame with target column.
    feature_cols : list[str]
        Columns to use as model features.
    target_col : str
        Binary target column name.
    config : dict
        XGBoost hyperparameter config (from config.yaml).
    model_path : str or Path
        If provided, save trained model to this path.

    Returns
    -------
    tuple[XGBClassifier, dict]
        Trained model and evaluation metrics dict.
    """
    if config is None:
        config = _default_config()

    df = df.copy().sort_values("timestamp") if "timestamp" in df.columns else df.copy()

    X = df[feature_cols].values
    y = df[target_col].values

    # Time-based split — no shuffling to preserve temporal structure
    n = len(X)
    split_idx = int(n * (1 - config.get("test_size", 0.2)))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Class imbalance weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)
    logger.info(f"Class balance — Negative: {n_neg}, Positive: {n_pos}, scale_pos_weight: {scale_pos_weight:.2f}")

    xgb_params = config.get("xgboost", {}).copy()
    xgb_params["scale_pos_weight"] = scale_pos_weight

    model = XGBClassifier(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    metrics = evaluate_model(model, X_test, y_test, feature_cols)

    if model_path is not None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
        logger.info(f"Model saved: {model_path}")

    return model, metrics


def evaluate_model(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
) -> dict[str, Any]:
    """
    Compute full evaluation suite.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1, roc_auc, confusion_matrix,
              classification_report, feature_importances.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "feature_importances": dict(zip(feature_cols, model.feature_importances_)),
    }

    logger.info(
        f"Evaluation — Accuracy: {metrics['accuracy']:.4f} | "
        f"F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}"
    )

    return metrics


def _default_config() -> dict:
    return {
        "test_size": 0.2,
        "xgboost": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": 42,
        },
    }