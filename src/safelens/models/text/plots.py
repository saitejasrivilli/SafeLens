"""Precision-recall curve and confusion-matrix plots for the baseline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay


def plot_precision_recall(y_true: list[int], y_prob: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(np.asarray(y_true), np.asarray(y_prob), ax=ax)
    ax.set_title("Precision-Recall (validation)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: list[int], y_pred: list[int], threshold: float, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        np.asarray(y_true), np.asarray(y_pred), ax=ax, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix (test, threshold={threshold})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
