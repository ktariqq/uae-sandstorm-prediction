"""
Inference module — standalone sandstorm risk predictor.

Can be used independently from the training pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

LOW_MAX = 0.30
MEDIUM_MAX = 0.60


@dataclass
class RiskOutput:
    risk_score: float
    risk_level: str
    feature_vector: dict

    def __str__(self) -> str:
        return (
            f"Risk Score: {self.risk_score:.3f} | "
            f"Risk Level: {self.risk_level}"
        )


class SandstormPredictor:
    """
    Standalone inference class for sandstorm risk prediction.

    Usage
    -----
    predictor = SandstormPredictor.from_saved("models/sandstorm_xgb.pkl")
    result = predictor.predict(feature_dict)
    print(result)
    """

    def __init__(
        self,
        model: XGBClassifier,
        feature_cols: list[str],
        low_max: float = LOW_MAX,
        medium_max: float = MEDIUM_MAX,
    ) -> None:
        self.model = model
        self.feature_cols = feature_cols
        self.low_max = low_max
        self.medium_max = medium_max

    @classmethod
    def from_saved(cls, path: str | Path) -> "SandstormPredictor":
        """Load predictor from saved joblib artifact."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        artifact = joblib.load(path)
        return cls(
            model=artifact["model"],
            feature_cols=artifact["feature_cols"],
        )

    def predict(self, features: dict | pd.DataFrame) -> RiskOutput:
        """
        Generate risk score and level for a single observation.

        Parameters
        ----------
        features : dict or pd.DataFrame
            Feature values keyed by feature name. Missing features filled with 0.

        Returns
        -------
        RiskOutput
        """
        if isinstance(features, dict):
            row = {col: features.get(col, 0.0) for col in self.feature_cols}
            X = np.array([[row[col] for col in self.feature_cols]])
        elif isinstance(features, pd.DataFrame):
            X = features[self.feature_cols].fillna(0).values
        else:
            raise TypeError("features must be a dict or pd.DataFrame.")

        prob = float(self.model.predict_proba(X)[0, 1])
        level = self._classify(prob)

        return RiskOutput(
            risk_score=prob,
            risk_level=level,
            feature_vector=dict(zip(self.feature_cols, X[0])),
        )

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for a full DataFrame.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Original DataFrame with 'risk_score' and 'risk_level' columns appended.
        """
        X = df[self.feature_cols].fillna(0).values
        probs = self.model.predict_proba(X)[:, 1]
        df = df.copy()
        df["risk_score"] = probs
        df["risk_level"] = [self._classify(p) for p in probs]
        return df

    def _classify(self, score: float) -> str:
        if score < self.low_max:
            return "LOW"
        elif score < self.medium_max:
            return "MEDIUM"
        else:
            return "HIGH"