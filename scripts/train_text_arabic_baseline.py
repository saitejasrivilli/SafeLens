#!/usr/bin/env python3
"""make text-arabic-train -- Phase 5B: frozen Arabic/multilingual text
encoder + classification head on Prop2Hate-Meme. NO image/filename access
anywhere in this pipeline.

Usage: python scripts/train_text_arabic_baseline.py --config configs/text_arabic_arabert.yaml

Pipeline (test set touched exactly once):
  1. load leakage-clean processed train (2,141) / unchanged dev (312) / test (606)
  2. extract frozen text-encoder embeddings for all three splits (no fine-tuning)
  3. train only a small classification head, model selection on dev PR-AUC
  4. evaluate at threshold 0.5 (fixed) on dev + test
  5. sweep decision thresholds on DEV ONLY, freeze selection
  6. evaluate the frozen threshold ONCE on TEST
  7. error analysis (text-only), latency benchmark, artifacts
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from safelens.data.multimodal.validation.report import load_valid_examples
from safelens.models.text.deberta.error_analysis import find_error_examples
from safelens.models.text.metrics import compute_metrics
from safelens.models.text.plots import plot_confusion_matrix, plot_precision_recall
from safelens.models.text.threshold import select_threshold, sweep_thresholds
from safelens.models.text_multilingual.artifacts import environment_metadata, save_artifacts
from safelens.models.text_multilingual.config import load_text_arabic_config
from safelens.models.text_multilingual.dataset import extract_embeddings
from safelens.models.text_multilingual.encoder import build_text_encoder
from safelens.models.text_multilingual.infer import (
    benchmark_encoder_latency,
    benchmark_end_to_end_latency,
    benchmark_head_latency,
    benchmark_tokenization_latency,
)
from safelens.models.vision.clip.head import ClassificationHead
from safelens.models.vision.clip.train import train_head
from safelens.utils.device import detect_device
from safelens.utils.logging import configure_logging, get_logger

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "multimodal" / "processed" / "prop2hate_meme"
MANIFEST_PATH = ROOT / "data" / "multimodal" / "manifests" / "prop2hate_meme_manifest.json"
RESULTS_ROOT = ROOT / "benchmarks" / "results" / "multimodal" / "text_only"


def main() -> None:
    configure_logging()
    log = get_logger("safelens.scripts.train_text_arabic_baseline")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_text_arabic_config(args.config)
    results_dir = RESULTS_ROOT / config.model_name
    device_info = detect_device()
    device = device_info.device
    log.info("[%s] device: %s (%s)", config.model_name, device, device_info.reason)

    examples_by_split = load_valid_examples(PROCESSED_DIR)
    train, dev, test = (
        examples_by_split["train"],
        examples_by_split["dev"],
        examples_by_split["test"],
    )
    log.info(
        "[%s] loaded (processed, text-only, NO image accessed): train=%d dev=%d test=%d",
        config.model_name,
        len(train),
        len(dev),
        len(test),
    )

    tokenizer, model = build_text_encoder(config)
    embed_dim = model.config.hidden_size
    param_count = sum(p.numel() for p in model.parameters())
    log.info(
        "[%s] text encoder frozen: %s @ %s (%d params, hidden=%d)",
        config.model_name,
        config.hf_model_name,
        config.hf_model_revision,
        param_count,
        embed_dim,
    )

    log.info("[%s] extracting frozen text embeddings (train/dev/test)...", config.model_name)
    t0 = time.perf_counter()
    train_emb, train_labels, _ = extract_embeddings(
        train, tokenizer, model, device, config.max_seq_length
    )
    dev_emb, dev_labels, dev_ids = extract_embeddings(
        dev, tokenizer, model, device, config.max_seq_length
    )
    test_emb, test_labels, test_ids = extract_embeddings(
        test, tokenizer, model, device, config.max_seq_length
    )
    embedding_extraction_time = time.perf_counter() - t0
    log.info("[%s] embedding extraction time: %.1fs", config.model_name, embedding_extraction_time)

    t0 = time.perf_counter()
    result = train_head(
        train_emb,
        train_labels,
        dev_emb,
        dev_labels,
        config.training,
        config.head,
        embed_dim=embed_dim,
    )
    training_time = time.perf_counter() - t0
    log.info(
        "[%s] training time: %.2fs, best_epoch=%d, epochs_run=%d, class_weights=%s",
        config.model_name,
        training_time,
        result.best_epoch,
        result.epochs_run,
        result.class_weights,
    )

    head = ClassificationHead(config.head, embed_dim=embed_dim)
    head.load_state_dict(result.best_state_dict)
    head.eval()

    with torch.no_grad():
        dev_probs = torch.softmax(head(dev_emb), dim=-1)[:, 1].numpy()
    dev_metrics_at_05 = compute_metrics(dev_labels.tolist(), dev_probs, 0.5)
    sweep = sweep_thresholds(
        dev_labels.tolist(), dev_probs.tolist(), config.decision_threshold.candidates
    )
    selected_threshold = select_threshold(sweep, config.decision_threshold.selection_metric)
    log.info(
        "[%s] selected decision threshold (dev-only): %.2f", config.model_name, selected_threshold
    )

    with torch.no_grad():
        test_probs = torch.softmax(head(test_emb), dim=-1)[:, 1].numpy()
    test_metrics_at_05 = compute_metrics(test_labels.tolist(), test_probs, 0.5)
    test_metrics_at_selected = compute_metrics(test_labels.tolist(), test_probs, selected_threshold)
    test_pred_at_selected = (test_probs >= selected_threshold).astype(int)

    error_analysis = find_error_examples(
        test_ids,
        [ex.text for ex in test],
        test_labels.tolist(),
        test_probs.tolist(),
        selected_threshold,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    plot_precision_recall(
        dev_labels.tolist(), dev_probs.tolist(), results_dir / "precision_recall_curve.png"
    )
    plot_confusion_matrix(
        test_labels.tolist(),
        test_pred_at_selected.tolist(),
        selected_threshold,
        results_dir / "confusion_matrix.png",
    )

    sample_text = test[0].text
    sample_encoding = tokenizer(
        sample_text,
        truncation=True,
        padding="max_length",
        max_length=config.max_seq_length,
        return_tensors="pt",
    )
    sample_embedding = test_emb[0:1]

    tokenization_bench = benchmark_tokenization_latency(
        tokenizer, sample_text, config.max_seq_length
    )
    encoder_bench = benchmark_encoder_latency(
        model, sample_encoding["input_ids"], sample_encoding["attention_mask"], device
    )
    head_bench = benchmark_head_latency(head, sample_embedding)
    end_to_end_bench = benchmark_end_to_end_latency(
        tokenizer, model, head, sample_text, config.max_seq_length, device
    )

    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}

    def _pos_rate(labels: torch.Tensor) -> float:
        return float(labels.float().mean()) if len(labels) else 0.0

    experiment = {
        "model_name": config.model_name,
        "model_version": config.model_version,
        "hf_model_name": config.hf_model_name,
        "hf_model_revision": config.hf_model_revision,
        "parameter_count": param_count,
        "embed_dim": embed_dim,
        "max_seq_length": config.max_seq_length,
        "config": config.model_dump(),
        "dataset_manifest_reference": str(MANIFEST_PATH),
        "dataset_manifest": manifest,
        "split_sizes": {"train": len(train), "dev": len(dev), "test": len(test)},
        "positive_rate": {
            "train": _pos_rate(train_labels),
            "dev": _pos_rate(dev_labels),
            "test": _pos_rate(test_labels),
        },
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
            "tokenization": tokenization_bench.__dict__,
            "text_encoder": encoder_bench.__dict__,
            "classification_head": head_bench.__dict__,
            "end_to_end": end_to_end_bench.__dict__,
        },
        "environment": environment_metadata(),
    }
    (results_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2, sort_keys=True, ensure_ascii=False)
    )
    log.info(
        "[%s] benchmark report written to %s", config.model_name, results_dir / "experiment.json"
    )

    artifact_path = save_artifacts(
        head_state_dict=result.best_state_dict,
        config=config,
        threshold=selected_threshold,
        dataset_manifest_path=MANIFEST_PATH,
        training_metadata=experiment["training_environment"],
    )
    log.info("[%s] model artifacts written to %s", config.model_name, artifact_path)

    log.info(
        "[%s] TEST @ threshold=0.5: f1=%.4f pr_auc=%.4f | "
        "TEST @ selected=%.2f: f1=%.4f pr_auc=%.4f",
        config.model_name,
        test_metrics_at_05["f1"],
        test_metrics_at_05["pr_auc"],
        selected_threshold,
        test_metrics_at_selected["f1"],
        test_metrics_at_selected["pr_auc"],
    )


if __name__ == "__main__":
    main()
