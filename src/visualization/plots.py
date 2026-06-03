"""
Visualization module — Purple–Pink Geospatial Intelligence Theme.

All plots use a unified dark aesthetic with purple/pink accent palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

# ── Theme palette ──────────────────────────────────────────────────────────
PALETTE = {
    "violet": "#7F00FF",
    "deep_purple": "#2E003E",
    "lavender": "#C8A2C8",
    "plum": "#8E4585",
    "hot_pink": "#FF007F",
    "shocking_pink": "#FF4DCC",
    "dark_pink": "#C2185B",
    "soft_pink": "#F8BBD0",
    "background": "#0D0010",
    "text": "#E8D5F5",
    "grid": "#3D1A5E",
}

ACCENT_SEQUENCE = [
    PALETTE["violet"],
    PALETTE["hot_pink"],
    PALETTE["lavender"],
    PALETTE["plum"],
    PALETTE["shocking_pink"],
    PALETTE["dark_pink"],
    PALETTE["soft_pink"],
]


def apply_theme() -> None:
    """Apply global matplotlib theme — call once at script start."""
    mpl.rcParams.update({
        "figure.facecolor": PALETTE["background"],
        "axes.facecolor": "#110018",
        "axes.edgecolor": PALETTE["grid"],
        "axes.labelcolor": PALETTE["text"],
        "axes.titlecolor": PALETTE["lavender"],
        "xtick.color": PALETTE["text"],
        "ytick.color": PALETTE["text"],
        "text.color": PALETTE["text"],
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        "legend.facecolor": "#1A0028",
        "legend.edgecolor": PALETTE["plum"],
        "font.family": "monospace",
        "axes.prop_cycle": mpl.cycler(color=ACCENT_SEQUENCE),
    })


def _save_or_show(fig: plt.Figure, path: str | Path | None) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Plot confusion matrix with theme styling."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(6, 5))

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "sandstorm_cm", [PALETTE["background"], PALETTE["violet"]]
    )
    im = ax.imshow(cm, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax)

    labels = ["No Event", "Sandstorm"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color=PALETTE["soft_pink"], fontsize=14, fontweight="bold"
            )

    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Confusion Matrix — Sandstorm Detection", fontsize=13, pad=12)

    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_roc_curve(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Plot ROC curve."""
    apply_theme()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color=PALETTE["violet"], lw=2, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=PALETTE["grid"], lw=1, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["violet"])

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Sandstorm Risk Classifier")
    ax.legend(loc="lower right")
    ax.grid(True)

    _save_or_show(fig, save_path)


def plot_feature_importance(
    importances: dict[str, float],
    top_n: int = 20,
    save_path: str | Path | None = None,
) -> None:
    """Plot top-N feature importances as horizontal bar chart."""
    apply_theme()

    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, scores = zip(*sorted_items)
    features = list(reversed(features))
    scores = list(reversed(scores))

    fig, ax = plt.subplots(figsize=(10, 0.4 * len(features) + 2))

    bar_colors = [
        PALETTE["violet"] if i % 3 == 0 else
        PALETTE["hot_pink"] if i % 3 == 1 else
        PALETTE["plum"]
        for i in range(len(features))
    ]

    bars = ax.barh(features, scores, color=bar_colors, edgecolor=PALETTE["grid"], linewidth=0.5)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}", va="center", ha="left",
            color=PALETTE["lavender"], fontsize=8
        )

    ax.set_xlabel("Feature Importance (Gain)", fontsize=10)
    ax.set_title("Feature Importance — XGBoost Sandstorm Model", fontsize=12, pad=12)
    ax.grid(axis="x", alpha=0.3)

    _save_or_show(fig, save_path)


def plot_risk_timeline(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    score_col: str = "risk_score",
    save_path: str | Path | None = None,
) -> None:
    """Plot risk score over time with threshold bands."""
    apply_theme()

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.fill_between(df[timestamp_col], 0, 0.30, alpha=0.08, color=PALETTE["lavender"], label="LOW zone")
    ax.fill_between(df[timestamp_col], 0.30, 0.60, alpha=0.08, color=PALETTE["plum"], label="MEDIUM zone")
    ax.fill_between(df[timestamp_col], 0.60, 1.0, alpha=0.08, color=PALETTE["hot_pink"], label="HIGH zone")

    ax.plot(df[timestamp_col], df[score_col], color=PALETTE["violet"], lw=0.8, alpha=0.9)

    ax.axhline(0.30, color=PALETTE["lavender"], lw=0.8, linestyle="--", alpha=0.6)
    ax.axhline(0.60, color=PALETTE["hot_pink"], lw=0.8, linestyle="--", alpha=0.6)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Risk Score")
    ax.set_xlabel("Timestamp")
    ax.set_title("Sandstorm Risk Score — Temporal Profile")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    _save_or_show(fig, save_path)