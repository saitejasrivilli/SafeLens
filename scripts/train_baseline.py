#!/usr/bin/env python3
"""make baseline-train -- reproducible TF-IDF + Logistic Regression baseline
on the exact Phase 2 processed civil_comments split.

Pipeline (test set touched exactly once, at the very end):
  1. load train/validation/test (Phase 2 output, unmodified)
  2. binarize continuous toxicity score -> ground-truth label
  3. fit TF-IDF on TRAIN TEXT ONLY
  4. fit Logistic Regression on TRAIN ONLY
  5. sweep decision thresholds on VALIDATION probabilities
  6. select + freeze threshold (validation-only criterion)
  7. evaluate ONCE on TEST using the frozen threshold
  8. save model artifacts + benchmark report
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from safelens.models.text.artifacts import environment_metadata, save_artifacts
from safelens.models.text.config import load_baseline_config
from safelens.models.text.data import binarize, load_split
from safelens.models.text.metrics import compute_metrics
from safelens.models.text.pipeline import fit_model, fit_vectorizer
from safelens.models.text.plots import plot_confusion_matrix, plot_precision_recall
from safelens.models.text.threshold import select_threshold, sweep_thresholds
from safelens.utils.logging import configure_logging, get_logger

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "manifests" / "civil_comments_manifest.json"
RESULTS_DIR = ROOT / "benchmarks" / "results" / "baseline"


def main() -> None:
    configure_logging()
    log = get_logger("safelens.scripts.train_baseline")

    config = load_baseline_config()
    log.info("config: %s", config.model_dump())

    train = load_split("train", config.label.target)
    val = load_split("validation", config.label.target)
    test = load_split("test", config.label.target)
    log.info(
        "loaded splits: train=%d validation=%d test=%d",
        len(train.texts),
        len(val.texts),
        len(test.texts),
    )

    y_train = binarize(train.scores, config.label.ground_truth_threshold)
    y_val = binarize(val.scores, config.label.ground_truth_threshold)
    y_test = binarize(test.scores, config.label.ground_truth_threshold)
    log.info(
        "positive rate @ ground_truth_threshold=%.2f: train=%.4f val=%.4f test=%.4f",
        config.label.ground_truth_threshold,
        sum(y_train) / len(y_train),
        sum(y_val) / len(y_val),
        sum(y_test) / len(y_test),
    )

    t0 = time.perf_counter()
    vectorizer = fit_vectorizer(train.texts, config.tfidf)  # fit on TRAIN ONLY
    X_train = vectorizer.transform(train.texts)
    model = fit_model(X_train, y_train, config.logistic_regression)
    training_time_s = time.perf_counter() - t0
    log.info("training time: %.3fs", training_time_s)

    X_val = vectorizer.transform(val.texts)  # transform only, never fit
    y_val_prob = model.predict_proba(X_val)[:, 1]

    sweep = sweep_thresholds(y_val, y_val_prob.tolist(), config.decision_threshold.candidates)
    threshold = select_threshold(sweep, config.decision_threshold.selection_metric)
    log.info("selected decision threshold (validation-only): %.2f", threshold)
    validation_metrics = compute_metrics(y_val, y_val_prob, threshold)

    # Test set touched exactly once, here, with the already-frozen threshold.
    X_test = vectorizer.transform(test.texts)  # transform only, never fit
    infer_t0 = time.perf_counter()
    y_test_prob = model.predict_proba(X_test)[:, 1]
    infer_elapsed = time.perf_counter() - infer_t0
    test_metrics = compute_metrics(y_test, y_test_prob, threshold)
    y_test_pred = (y_test_prob >= threshold).astype(int)

    inference_throughput = len(test.texts) / infer_elapsed if infer_elapsed > 0 else float("inf")
    inference_latency_ms_per_example = (infer_elapsed / len(test.texts)) * 1000

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_precision_recall(y_val, y_val_prob.tolist(), RESULTS_DIR / "precision_recall_curve.png")
    plot_confusion_matrix(
        y_test, y_test_pred.tolist(), threshold, RESULTS_DIR / "confusion_matrix.png"
    )

    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}

    experiment = {
        "model_name": config.model_name,
        "model_version": config.model_version,
        "config": config.model_dump(),
        "dataset_manifest_reference": str(MANIFEST_PATH),
        "dataset_version": train.dataset_version,
        "dataset_manifest": manifest,
        "split_sizes": {
            "train": len(train.texts),
            "validation": len(val.texts),
            "test": len(test.texts),
        },
        "positive_rate": {
            "train": sum(y_train) / len(y_train),
            "validation": sum(y_val) / len(y_val),
            "test": sum(y_test) / len(y_test),
        },
        "threshold_sweep": sweep,
        "selected_threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "training_time_seconds": training_time_s,
        "inference": {
            "environment": "Measured locally on Apple M2",
            "test_set_size": len(test.texts),
            "total_inference_seconds": infer_elapsed,
            "throughput_examples_per_second": inference_throughput,
            "latency_ms_per_example": inference_latency_ms_per_example,
        },
        "environment": environment_metadata(),
    }
    (RESULTS_DIR / "experiment.json").write_text(json.dumps(experiment, indent=2, sort_keys=True))
    log.info("benchmark report written to %s", RESULTS_DIR / "experiment.json")

    artifact_path = save_artifacts(
        model_version=config.model_version,
        vectorizer=vectorizer,
        model=model,
        config=config,
        dataset_version=train.dataset_version,
        dataset_manifest_path=MANIFEST_PATH,
    )
    log.info("model artifacts written to %s", artifact_path)

    log.info(
        "TEST metrics @ threshold=%.2f: precision=%.4f recall=%.4f f1=%.4f "
        "pr_auc=%.4f fpr=%.4f fnr=%.4f",
        threshold,
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
        test_metrics["pr_auc"],
        test_metrics["false_positive_rate"],
        test_metrics["false_negative_rate"],
    )


if __name__ == "__main__":
    main()
