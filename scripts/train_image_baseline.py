#!/usr/bin/env python3
"""make image-baseline-train -- Phase 5A: frozen-CLIP image-only baseline
on Prop2Hate-Meme. NO access to the text field anywhere in this pipeline.

Pipeline (test set touched exactly once):
  1. load leakage-clean processed train (2,141) / unchanged dev (312) / test (606)
  2. extract frozen CLIP image embeddings for all three splits (no fine-tuning)
  3. train only a small classification head, model selection on dev PR-AUC
  4. evaluate at threshold 0.5 (fixed) on dev + test
  5. sweep decision thresholds on DEV ONLY, freeze selection
  6. evaluate the frozen threshold ONCE on TEST
  7. error analysis (image-only, no text), latency benchmark, artifacts
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import torch
from PIL import Image

from safelens.data.multimodal.validation.report import load_valid_examples
from safelens.models.text.metrics import compute_metrics
from safelens.models.text.plots import plot_confusion_matrix, plot_precision_recall
from safelens.models.text.threshold import select_threshold, sweep_thresholds
from safelens.models.vision.clip.artifacts import environment_metadata, save_artifacts
from safelens.models.vision.clip.config import load_image_baseline_config
from safelens.models.vision.clip.dataset import extract_embeddings
from safelens.models.vision.clip.encoder import build_clip_encoder
from safelens.models.vision.clip.error_analysis import find_error_examples
from safelens.models.vision.clip.head import ClassificationHead
from safelens.models.vision.clip.infer import (
    benchmark_encoder_latency,
    benchmark_end_to_end_latency,
    benchmark_head_latency,
    benchmark_preprocessing_latency,
)
from safelens.models.vision.clip.train import train_head
from safelens.utils.device import detect_device
from safelens.utils.logging import configure_logging, get_logger

ROOT = Path(__file__).resolve().parents[1]
RAW_IMAGE_ROOT = ROOT / "data" / "multimodal" / "raw" / "prop2hate_meme"
PROCESSED_DIR = ROOT / "data" / "multimodal" / "processed" / "prop2hate_meme"
MANIFEST_PATH = ROOT / "data" / "multimodal" / "manifests" / "prop2hate_meme_manifest.json"
RESULTS_DIR = ROOT / "benchmarks" / "results" / "multimodal" / "image_only"


def main() -> None:
    configure_logging()
    log = get_logger("safelens.scripts.train_image_baseline")

    config = load_image_baseline_config()
    device_info = detect_device()
    device = device_info.device
    log.info("device: %s (%s)", device, device_info.reason)

    examples_by_split = load_valid_examples(PROCESSED_DIR)
    train, dev, test = (
        examples_by_split["train"],
        examples_by_split["dev"],
        examples_by_split["test"],
    )
    log.info(
        "loaded (processed, image-only, NO text accessed): train=%d dev=%d test=%d",
        len(train),
        len(dev),
        len(test),
    )

    processor, model = build_clip_encoder(config)
    log.info("CLIP encoder frozen: %s @ %s", config.clip_model_name, config.clip_revision)

    log.info("extracting frozen CLIP embeddings (train/dev/test)...")
    t0 = time.perf_counter()
    train_emb, train_labels, _ = extract_embeddings(train, RAW_IMAGE_ROOT, processor, model, device)
    dev_emb, dev_labels, dev_ids = extract_embeddings(dev, RAW_IMAGE_ROOT, processor, model, device)
    test_emb, test_labels, test_ids = extract_embeddings(
        test, RAW_IMAGE_ROOT, processor, model, device
    )
    embedding_extraction_time = time.perf_counter() - t0
    log.info("embedding extraction time: %.1fs", embedding_extraction_time)

    t0 = time.perf_counter()
    result = train_head(train_emb, train_labels, dev_emb, dev_labels, config.training, config.head)
    training_time = time.perf_counter() - t0
    log.info(
        "training time: %.2fs, best_epoch=%d, epochs_run=%d, class_weights=%s",
        training_time,
        result.best_epoch,
        result.epochs_run,
        result.class_weights,
    )

    head = ClassificationHead(config.head)
    head.load_state_dict(result.best_state_dict)
    head.eval()

    with torch.no_grad():
        dev_probs = torch.softmax(head(dev_emb), dim=-1)[:, 1].numpy()
    dev_metrics_at_05 = compute_metrics(dev_labels.tolist(), dev_probs, 0.5)
    sweep = sweep_thresholds(
        dev_labels.tolist(), dev_probs.tolist(), config.decision_threshold.candidates
    )
    selected_threshold = select_threshold(sweep, config.decision_threshold.selection_metric)
    log.info("selected decision threshold (dev-only): %.2f", selected_threshold)

    # Test set touched exactly once, here.
    with torch.no_grad():
        test_probs = torch.softmax(head(test_emb), dim=-1)[:, 1].numpy()
    test_metrics_at_05 = compute_metrics(test_labels.tolist(), test_probs, 0.5)
    test_metrics_at_selected = compute_metrics(test_labels.tolist(), test_probs, selected_threshold)
    test_pred_at_selected = (test_probs >= selected_threshold).astype(int)

    error_analysis = find_error_examples(
        test_ids,
        [ex.image_path for ex in test],
        test_labels.tolist(),
        test_probs.tolist(),
        selected_threshold,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_precision_recall(
        dev_labels.tolist(), dev_probs.tolist(), RESULTS_DIR / "precision_recall_curve.png"
    )
    plot_confusion_matrix(
        test_labels.tolist(),
        test_pred_at_selected.tolist(),
        selected_threshold,
        RESULTS_DIR / "confusion_matrix.png",
    )

    # Latency benchmark, stage-by-stage, batch size 1.
    sample_example = test[0]
    sample_image_path = RAW_IMAGE_ROOT / sample_example.image_path
    sample_image = Image.open(sample_image_path).convert("RGB")
    sample_pixel_values = processor(images=sample_image, return_tensors="pt")["pixel_values"]
    sample_embedding = test_emb[0:1]

    preprocessing_bench = benchmark_preprocessing_latency(processor, sample_image)
    encoder_bench = benchmark_encoder_latency(model, sample_pixel_values, device)
    head_bench = benchmark_head_latency(head, sample_embedding)
    end_to_end_bench = benchmark_end_to_end_latency(
        processor, model, head, sample_image_path, device
    )

    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}

    def _pos_rate(labels: torch.Tensor) -> float:
        return float(labels.float().mean()) if len(labels) else 0.0

    experiment = {
        "model_name": config.model_name,
        "model_version": config.model_version,
        "clip_model_name": config.clip_model_name,
        "clip_revision": config.clip_revision,
        "config": config.model_dump(),
        "dataset_manifest_reference": str(MANIFEST_PATH),
        "dataset_manifest": manifest,
        "split_sizes": {"train": len(train), "dev": len(dev), "test": len(test)},
        "positive_rate": {
            "train": _pos_rate(train_labels),
            "dev": _pos_rate(dev_labels),
            "test": _pos_rate(test_labels),
        },
        "distribution_shift_note": (
            "test positive rate differs substantially from train/dev (measured, not "
            "explained by official docs) -- see docs/multimodal_design.md sec 10b. "
            "Threshold selected on dev only; test evaluated once, not used for selection."
        ),
        "training_environment": {
            "os": platform.platform(),
            "device": device,
            "device_reason": device_info.reason,
            "embedding_extraction_seconds": embedding_extraction_time,
            "head_training_seconds": training_time,
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "class_weights": result.class_weights,
        },
        "dev_metrics_at_threshold_0.5": dev_metrics_at_05,
        "threshold_sweep_dev_only": sweep,
        "selected_threshold": selected_threshold,
        "test_metrics_at_threshold_0.5": test_metrics_at_05,
        "test_metrics_at_selected_threshold": test_metrics_at_selected,
        "error_analysis": error_analysis,
        "inference_benchmark": {
            "environment": f"Measured on {platform.platform()}, device={device}",
            "preprocessing": preprocessing_bench.__dict__,
            "clip_encoder": encoder_bench.__dict__,
            "classification_head": head_bench.__dict__,
            "end_to_end": end_to_end_bench.__dict__,
        },
        "environment": environment_metadata(),
    }
    (RESULTS_DIR / "experiment.json").write_text(json.dumps(experiment, indent=2, sort_keys=True))
    log.info("benchmark report written to %s", RESULTS_DIR / "experiment.json")

    artifact_path = save_artifacts(
        model_version=config.model_version,
        head_state_dict=result.best_state_dict,
        config=config,
        threshold=selected_threshold,
        dataset_manifest_path=MANIFEST_PATH,
        training_metadata=experiment["training_environment"],
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
