"""Decision-threshold sweep and selection. Must only ever be run against the
validation set — never the test set."""

from __future__ import annotations

from typing import Any

from safelens.models.text.metrics import compute_metrics


def sweep_thresholds(
    y_true: list[int], y_prob: list[float], candidates: list[float]
) -> list[dict[str, Any]]:
    return [compute_metrics(y_true, y_prob, t) for t in candidates]


def select_threshold(sweep: list[dict[str, Any]], selection_metric: str) -> float:
    """Picks the candidate with the highest value of `selection_metric`.
    Ties broken by the smallest threshold (more conservative / higher
    recall), for determinism."""
    best = max(sweep, key=lambda row: (row[selection_metric], -row["threshold"]))
    return float(best["threshold"])
