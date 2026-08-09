#!/usr/bin/env python3
"""make deberta-train -- fine-tune microsoft/deberta-v3-small on the exact
Phase 2 processed civil_comments split, directly comparable to the frozen
Phase 3 TF-IDF + Logistic Regression baseline.

Pipeline (test set touched exactly once):
  1. load train/validation/test (Phase 2 output, unmodified)
  2. binarize continuous toxicity score -> ground-truth label (same threshold as Phase 3)
  3. tokenize with the official DeBERTa-v3 tokenizer (deterministic)
  4. fine-tune with class-weighted loss, model selection on validation PR-AUC
  5. evaluate at threshold 0.5 (direct baseline comparison) on validation + test
  6. sweep decision thresholds on VALIDATION probabilities, freeze selection
  7. evaluate the frozen threshold ONCE on TEST
  8. error analysis, inference benchmark, comparison table, artifacts
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from safelens.models.text.data import binarize, load_split
from safelens.models.text.deberta.artifacts import environment_metadata, save_artifacts
from safelens.models.text.deberta.config import load_deberta_config
from safelens.models.text.deberta.dataset import ToxicityDataset
from safelens.models.text.deberta.error_analysis import find_error_examples
from safelens.models.text.deberta.infer import (
    benchmark_end_to_end_latency,
    benchmark_model_only_latency,
    predict_proba,
)
from safelens.models.text.deberta.train import (
    WeightedTrainer,
    build_compute_metrics_fn,
    build_model_and_tokenizer,
    compute_class_weights,
)
from safelens.models.text.metrics import compute_metrics
from safelens.models.text.plots import plot_confusion_matrix, plot_precision_recall
from safelens.models.text.threshold import select_threshold, sweep_thresholds
from safelens.utils.device import detect_device
from safelens.utils.logging import configure_logging, get_logger

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "manifests" / "civil_comments_manifest.json"
RESULTS_DIR = ROOT / "benchmarks" / "results" / "deberta"
BASELINE_EXPERIMENT_PATH = ROOT / "benchmarks" / "results" / "baseline" / "experiment.json"
CHECKPOINT_DIR = ROOT / "models" / "text" / "deberta" / "checkpoints"  # gitignored scratch space


def main() -> None:
    configure_logging()
    log = get_logger("safelens.scripts.train_deberta")

    config = load_deberta_config()
    device_info = detect_device()
    log.info("device: %s (%s)", device_info.device, device_info.reason)
    if not device_info.torch_available:
        raise SystemExit("torch not available in this environment -- cannot run Phase 4.")

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

    tokenizer, model = build_model_and_tokenizer(config)
    param_count = sum(p.numel() for p in model.parameters())
    log.info("model params: %d", param_count)

    train_ds = ToxicityDataset(
        train.texts, y_train, train.content_ids, tokenizer, config.tokenizer.max_seq_length
    )
    val_ds = ToxicityDataset(
        val.texts, y_val, val.content_ids, tokenizer, config.tokenizer.max_seq_length
    )
    test_ds = ToxicityDataset(
        test.texts, y_test, test.content_ids, tokenizer, config.tokenizer.max_seq_length
    )

    class_weights = compute_class_weights(y_train) if config.training.use_class_weighting else None
    log.info("class weights: %s", class_weights)

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.train_batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        max_grad_norm=config.training.max_grad_norm,
        seed=config.training.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=config.training.model_selection_metric,
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
        disable_tqdm=False,
    )

    trainer_cls = WeightedTrainer if class_weights is not None else Trainer
    trainer_kwargs = {"class_weights": class_weights} if class_weights is not None else {}
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=build_compute_metrics_fn(),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config.training.early_stopping_patience)
        ],
        **trainer_kwargs,
    )

    t0 = time.perf_counter()
    trainer.train()
    training_time_s = time.perf_counter() - t0
    log.info("training time: %.1fs", training_time_s)

    actual_device = str(next(model.parameters()).device)
    log.info("model actually trained on device: %s", actual_device)

    # Validation probabilities (drives threshold sweep + model-selection reporting).
    val_pred = trainer.predict(val_ds)
    val_probs = torch.softmax(torch.from_numpy(val_pred.predictions), dim=-1)[:, 1].numpy()

    validation_metrics_at_05 = compute_metrics(y_val, val_probs, 0.5)
    sweep = sweep_thresholds(y_val, val_probs.tolist(), config.decision_threshold.candidates)
    selected_threshold = select_threshold(sweep, config.decision_threshold.selection_metric)
    log.info("selected decision threshold (validation-only): %.2f", selected_threshold)

    # Test set touched exactly once, here.
    test_pred = trainer.predict(test_ds)
    test_probs = torch.softmax(torch.from_numpy(test_pred.predictions), dim=-1)[:, 1].numpy()

    test_metrics_at_05 = compute_metrics(y_test, test_probs, 0.5)
    test_metrics_at_selected = compute_metrics(y_test, test_probs, selected_threshold)
    test_pred_at_selected = (test_probs >= selected_threshold).astype(int)

    error_analysis = find_error_examples(
        test.content_ids, test.texts, y_test, test_probs.tolist(), selected_threshold
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_precision_recall(y_val, val_probs.tolist(), RESULTS_DIR / "precision_recall_curve.png")
    plot_confusion_matrix(
        y_test,
        test_pred_at_selected.tolist(),
        selected_threshold,
        RESULTS_DIR / "confusion_matrix.png",
    )

    # Inference benchmark (model actually on `actual_device`, local M2 dev machine).
    sample_text = test.texts[0]
    model_only_bench = benchmark_model_only_latency(
        model, tokenizer, sample_text, config.tokenizer.max_seq_length, actual_device
    )
    end_to_end_bench = benchmark_end_to_end_latency(
        model, tokenizer, sample_text, config.tokenizer.max_seq_length, actual_device
    )
    throughput_t0 = time.perf_counter()
    predict_proba(
        model, tokenizer, test.texts, config.tokenizer.max_seq_length, actual_device, batch_size=32
    )
    throughput_elapsed = time.perf_counter() - throughput_t0

    baseline_experiment = (
        json.loads(BASELINE_EXPERIMENT_PATH.read_text())
        if BASELINE_EXPERIMENT_PATH.exists()
        else {}
    )
    baseline_test_metrics = baseline_experiment.get("test_metrics", {})

    def _delta(key: str, deberta_value: float) -> dict[str, float]:
        baseline_value = baseline_test_metrics.get(key)
        if baseline_value is None:
            return {}
        absolute = deberta_value - baseline_value
        relative = (absolute / baseline_value) if baseline_value else float("nan")
        return {
            "baseline": baseline_value,
            "deberta": deberta_value,
            "absolute_delta": absolute,
            "relative_delta": relative,
        }

    comparison = {
        "f1": _delta("f1", test_metrics_at_selected["f1"]),
        "pr_auc": _delta("pr_auc", test_metrics_at_selected["pr_auc"]),
        "precision": _delta("precision", test_metrics_at_selected["precision"]),
        "recall": _delta("recall", test_metrics_at_selected["recall"]),
        "false_positive_rate": _delta(
            "false_positive_rate", test_metrics_at_selected["false_positive_rate"]
        ),
        "false_negative_rate": _delta(
            "false_negative_rate", test_metrics_at_selected["false_negative_rate"]
        ),
    }

    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}

    experiment = {
        "model_name": config.model_name,
        "model_version": config.model_version,
        "hf_model_name": config.hf_model_name,
        "hf_model_revision": config.hf_model_revision,
        "parameter_count": param_count,
        "max_seq_length": config.tokenizer.max_seq_length,
        "config": config.model_dump(),
        "dataset_manifest_reference": str(MANIFEST_PATH),
        "dataset_version": train.dataset_version,
        "dataset_manifest": manifest,
        "split_sizes": {
            "train": len(train.texts),
            "validation": len(val.texts),
            "test": len(test.texts),
        },
        "training_environment": {
            "os": platform.platform(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": actual_device,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "training_duration_seconds": training_time_s,
            "train_batch_size": config.training.train_batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "effective_batch_size": config.training.train_batch_size
            * config.training.gradient_accumulation_steps,
            "max_seq_length": config.tokenizer.max_seq_length,
            "epochs_configured": config.training.epochs,
        },
        "validation_metrics_at_threshold_0.5": validation_metrics_at_05,
        "threshold_sweep": sweep,
        "selected_threshold": selected_threshold,
        "test_metrics_at_threshold_0.5": test_metrics_at_05,
        "test_metrics_at_selected_threshold": test_metrics_at_selected,
        "comparison_vs_baseline": comparison,
        "error_analysis": error_analysis,
        "inference_benchmark": {
            "environment": "Measured locally on Apple M2",
            "model_only": model_only_bench.__dict__,
            "end_to_end": end_to_end_bench.__dict__,
            "batch_throughput": {
                "batch_size": 32,
                "test_set_size": len(test.texts),
                "total_seconds": throughput_elapsed,
                "throughput_examples_per_second": len(test.texts) / throughput_elapsed,
            },
        },
        "environment": environment_metadata(),
    }
    (RESULTS_DIR / "experiment.json").write_text(json.dumps(experiment, indent=2, sort_keys=True))
    log.info("benchmark report written to %s", RESULTS_DIR / "experiment.json")

    artifact_path = save_artifacts(
        model_version=config.model_version,
        model=model,
        tokenizer=tokenizer,
        config=config,
        threshold=selected_threshold,
        dataset_version=train.dataset_version,
        dataset_manifest_path=MANIFEST_PATH,
    )
    log.info("model artifacts written to %s", artifact_path)

    log.info(
        "TEST @ threshold=0.5: f1=%.4f pr_auc=%.4f | TEST @ selected=%.2f: f1=%.4f pr_auc=%.4f",
        test_metrics_at_05["f1"],
        test_metrics_at_05["pr_auc"],
        selected_threshold,
        test_metrics_at_selected["f1"],
        test_metrics_at_selected["pr_auc"],
    )


if __name__ == "__main__":
    main()
