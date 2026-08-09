import math

from safelens.models.text.metrics import compute_metrics
from safelens.models.text.threshold import select_threshold, sweep_thresholds

Y_TRUE = [0, 0, 0, 1, 1, 1, 1, 0, 1, 0]
Y_PROB = [0.05, 0.2, 0.4, 0.9, 0.8, 0.55, 0.3, 0.1, 0.6, 0.7]


def test_compute_metrics_known_case():
    # threshold 0.5 -> predictions: [0,0,0,1,1,1,0,0,1,1]
    metrics = compute_metrics(Y_TRUE, Y_PROB, threshold=0.5)
    cm = metrics["confusion_matrix"]
    assert cm["true_positive"] + cm["false_negative"] == sum(Y_TRUE)
    assert cm["true_negative"] + cm["false_positive"] == len(Y_TRUE) - sum(Y_TRUE)
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["false_positive_rate"] <= 1.0
    assert 0.0 <= metrics["false_negative_rate"] <= 1.0
    assert not math.isnan(metrics["pr_auc"])
    assert metrics["n"] == len(Y_TRUE)


def test_metrics_no_positives_no_nan_crash():
    y_true = [0, 0, 0, 0]
    y_prob = [0.1, 0.2, 0.05, 0.3]
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["false_positive_rate"] == 0.0
    assert math.isnan(metrics["pr_auc"])  # undefined with a single class present


def test_sweep_thresholds_covers_all_candidates():
    candidates = [0.1, 0.3, 0.5, 0.7, 0.9]
    sweep = sweep_thresholds(Y_TRUE, Y_PROB, candidates)
    assert [row["threshold"] for row in sweep] == candidates


def test_select_threshold_picks_best_f1():
    candidates = [0.1, 0.3, 0.5, 0.7, 0.9]
    sweep = sweep_thresholds(Y_TRUE, Y_PROB, candidates)
    selected = select_threshold(sweep, "f1")
    best_row = max(sweep, key=lambda r: r["f1"])
    assert selected == best_row["threshold"]


def test_select_threshold_deterministic_tiebreak():
    # two rows tied on the selection metric -> smaller threshold wins
    sweep = [
        {"threshold": 0.3, "f1": 0.8},
        {"threshold": 0.6, "f1": 0.8},
        {"threshold": 0.9, "f1": 0.5},
    ]
    assert select_threshold(sweep, "f1") == 0.3
