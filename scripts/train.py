"""
Training entry point.

Usage:
  python scripts/train.py
  python scripts/train.py --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.helpers import load_config, setup_logging
from src.data.merra2_loader import load_merra2
from src.data.ncms_loader import load_ncms, merge_datasets
from src.features.engineering import engineer_features, get_feature_columns
from src.features.labeling import label_sandstorm_events
from src.model.train import train_model
from src.visualization.plots import (
    apply_theme,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
)
from sklearn.metrics import roc_curve
import numpy as np
import logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train sandstorm XGBoost model")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)

    logger.info("=== UAE Sandstorm Prediction System — Training ===")

    # Load data
    merra2 = load_merra2(cfg["paths"]["raw_merra2"])
    ncms = load_ncms(cfg["paths"]["raw_ncms"])
    merged = merge_datasets(merra2, ncms)

    # Label
    label_cfg = cfg["labeling"]
    labeled = label_sandstorm_events(
        merged,
        visibility_threshold=label_cfg["visibility_threshold"],
        wind_speed_threshold=label_cfg["wind_speed_threshold"],
        require_both=label_cfg["require_both"],
        target_col=cfg["data"]["target_col"],
    )

    # Feature engineering
    feat_cfg = cfg["features"]
    featured = engineer_features(
        labeled,
        lag_hours=feat_cfg["lag_hours"],
        rolling_windows=feat_cfg["rolling_windows"],
        cyclical=feat_cfg["cyclical_features"],
    )

    target_col = cfg["data"]["target_col"]

    # ---------------- SAFE TARGET FIX ----------------
    y = featured[target_col]

    # force numeric
    y = np.asarray(y)

    # hard binary conversion (CRITICAL FIX)
    y = (y > 0.5).astype(int)

    featured[target_col] = y
    # --------------------------------------------------

    feature_cols = get_feature_columns(featured, target_col)

    # Train
    model, metrics = train_model(
        featured,
        feature_cols=feature_cols,
        target_col=target_col,
        config={
            "test_size": cfg["data"]["test_size"],
            "xgboost": cfg["model"]["xgboost"],
        },
        model_path=cfg["paths"]["model_output"],
    )

    # Report
    logger.info("\n" + metrics["classification_report"])
    logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    # Save evaluation plots
    report_dir = Path(cfg["paths"]["report_dir"])
    apply_theme()

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        save_path=report_dir / "confusion_matrix.png"
    )
    logger.info("Confusion matrix saved.")

    # ROC curve
    n = len(featured)
    split_idx = int(n * (1 - cfg["data"]["test_size"]))

    X = featured[feature_cols].values
    y = featured[target_col].values

    # extra safety (prevents float leakage again)
    y = (y > 0).astype(int)

    X_test, y_test = X[split_idx:], y[split_idx:]
    y_prob = model.predict_proba(X_test)[:, 1]

    plot_roc_curve(
        y_test,
        y_prob,
        save_path=report_dir / "roc_curve.png"
    )
    logger.info("ROC curve saved.")

    plot_feature_importance(
        metrics["feature_importances"],
        save_path=report_dir / "feature_importance.png"
    )
    logger.info("Feature importance plot saved.")

    logger.info("=== Training complete ===")


if __name__ == "__main__":
    main()