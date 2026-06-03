"""
Inference entry point.

Usage:
  python scripts/predict.py --input data/processed/features.csv --output reports/predictions.csv
  python scripts/predict.py --single  (manual input mode)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import logging

from src.utils.helpers import load_config, setup_logging
from src.model.inference import SandstormPredictor

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run sandstorm risk inference")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", default=None, help="Path to input feature CSV")
    parser.add_argument("--output", default="reports/predictions.csv")
    parser.add_argument("--single", action="store_true", help="Manual single-observation mode")
    return parser.parse_args()


def single_observation_mode(predictor: SandstormPredictor) -> None:
    print("\n[Single Observation Mode]")
    print("Enter values for key features (press Enter to use 0.0):\n")

    prompt_features = [
        "wind_speed", "humidity", "surface_pressure",
        "aerosol_optical_depth", "wind_speed_lag_1h",
        "wind_speed_lag_3h", "wind_speed_roll_6h",
        "humidity_pressure_ratio",
    ]

    inputs = {}
    for feat in prompt_features:
        raw = input(f"  {feat}: ").strip()
        inputs[feat] = float(raw) if raw else 0.0

    result = predictor.predict(inputs)
    print(f"\n{'='*50}")
    print(f"  RISK SCORE : {result.risk_score:.4f}")
    print(f"  RISK LEVEL : {result.risk_level}")
    print(f"{'='*50}\n")


def main():
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)

    logger.info("=== UAE Sandstorm Prediction System — Inference ===")

    predictor = SandstormPredictor.from_saved(cfg["paths"]["model_output"])

    if args.single:
        single_observation_mode(predictor)
        return

    if args.input is None:
        logger.error("Provide --input path or use --single mode.")
        sys.exit(1)

    df = pd.read_csv(args.input)
    results = predictor.predict_batch(df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    logger.info(f"Predictions written to: {output_path}")

    level_counts = results["risk_level"].value_counts()
    for level, count in level_counts.items():
        logger.info(f"  {level}: {count} records ({count/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()