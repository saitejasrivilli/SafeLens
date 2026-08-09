import math

import numpy as np
from transformers import EvalPrediction

from safelens.models.text.deberta.train import build_compute_metrics_fn, compute_class_weights


def test_compute_class_weights_balances_rare_positive_class():
    labels = [0] * 95 + [1] * 5  # 5% positive, same imbalance as civil_comments
    weights = compute_class_weights(labels)
    assert weights.shape == (2,)
    assert weights[1] > weights[0]  # rare positive class gets more weight


def test_compute_class_weights_balanced_input():
    labels = [0, 1, 0, 1]
    weights = compute_class_weights(labels)
    assert math.isclose(weights[0].item(), weights[1].item(), rel_tol=1e-6)


def test_compute_metrics_fn_returns_flat_scalars():
    logits = np.array([[2.0, 0.1], [0.1, 2.0], [1.5, 0.2], [0.2, 1.8]], dtype=np.float32)
    labels = np.array([0, 1, 0, 1])
    eval_pred = EvalPrediction(predictions=logits, label_ids=labels)
    fn = build_compute_metrics_fn()
    metrics = fn(eval_pred)
    for key in (
        "f1",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "false_positive_rate",
        "false_negative_rate",
    ):
        assert key in metrics
        assert isinstance(metrics[key], float)
        assert not math.isnan(metrics[key])


def test_compute_metrics_fn_handles_tuple_predictions():
    logits = np.array([[2.0, 0.1], [0.1, 2.0]], dtype=np.float32)
    labels = np.array([0, 1])
    eval_pred = EvalPrediction(predictions=(logits, None), label_ids=labels)
    fn = build_compute_metrics_fn()
    metrics = fn(eval_pred)
    assert 0.0 <= metrics["precision"] <= 1.0
