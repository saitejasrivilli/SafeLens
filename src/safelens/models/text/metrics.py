"""Moderation-relevant metrics. Accuracy is computed but never the headline."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: list[int] | np.ndarray, y_prob: list[float] | np.ndarray, threshold: float
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    has_both_classes = len(set(y_true.tolist())) > 1

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "pr_auc": average_precision_score(y_true, y_prob) if has_both_classes else float("nan"),
        "roc_auc": roc_auc_score(y_true, y_prob) if has_both_classes else float("nan"),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "positive_prediction_rate": float(y_pred.mean()),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "n": int(len(y_true)),
    }
