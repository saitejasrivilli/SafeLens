#!/usr/bin/env python3
"""make multimodal-fusion-train -- Phase 5C: frozen CLIP + frozen AraBERT ->
concatenation -> fusion MLP head on Prop2Hate-Meme.

Uses the EXACT checkpoints already approved in Phase 5A (CLIP) and Phase 5B
(AraBERT). Only the fusion head is trained. Reloads the already-trained
Phase 5A/5B standalone heads (not retrained) purely for the mandatory
complementarity analysis.

Pipeline (test set touched exactly once for threshold-bearing metrics):
  1. load leakage-clean processed train (2,141) / unchanged dev (312) / test (606)
  2. extract frozen CLIP image + frozen AraBERT text embeddings, concatenate
  3. train only the fusion head, model selection on dev PR-AUC
  4. evaluate at threshold 0.5 (fixed, primary) on dev + test
  5. sweep decision thresholds on DEV ONLY (secondary), freeze selection, evaluate test once
  6. ablation: fusion model with one modality replaced by its train-mean vector
  7. complementarity analysis vs. the standalone Phase 5A/5B models
  8. error analysis (text + image), latency benchmark, artifacts
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import torch

from safelens.data.multimodal.validation.report import load_valid_examples
from safelens.models.multimodal.artifacts import environment_metadata, save_artifacts
from safelens.models.multimodal.complementarity import analyze_complementarity
from safelens.models.multimodal.config import load_fusion_config
from safelens.models.multimodal.dataset import extract_fused_embeddings
from safelens.models.multimodal.error_analysis import find_error_examples
from safelens.models.multimodal.infer import (
    benchmark_fusion_end_to_end_latency,
    benchmark_fusion_head_latency,
)
from safelens.models.multimodal.missing_modality import (
    build_image_only_input,
    build_text_only_input,
    compute_modality_means,
)
from safelens.models.text.metrics import compute_metrics
from safelens.models.text.plots import plot_confusion_matrix, plot_precision_recall
from safelens.models.text.threshold import select_threshold, sweep_thresholds
from safelens.models.text_multilingual.artifacts import load_artifacts as load_text_artifacts
from safelens.models.text_multilingual.config import TextArabicConfig
from safelens.models.text_multilingual.encoder import build_text_encoder
from safelens.models.text_multilingual.infer import (
    benchmark_encoder_latency as benchmark_text_encoder_latency,
)
from safelens.models.text_multilingual.infer import (
    benchmark_tokenization_latency,
)
from safelens.models.vision.clip.artifacts import load_artifacts as load_clip_artifacts
from safelens.models.vision.clip.encoder import build_clip_encoder
from safelens.models.vision.clip.head import ClassificationHead
from safelens.models.vision.clip.infer import (
    benchmark_encoder_latency,
    benchmark_preprocessing_latency,
)
from safelens.models.vision.clip.train import train_head
from safelens.utils.device import detect_device
from safelens.utils.logging import configure_logging, get_logger

ROOT = Path(__file__).resolve().parents[1]
RAW_IMAGE_ROOT = ROOT / "data" / "multimodal" / "raw" / "prop2hate_meme"
PROCESSED_DIR = ROOT / "data" / "multimodal" / "processed" / "prop2hate_meme"
MANIFEST_PATH = ROOT / "data" / "multimodal" / "manifests" / "prop2hate_meme_manifest.json"
RESULTS_DIR = ROOT / "benchmarks" / "results" / "multimodal" / "fusion"


def _pos_rate(labels: torch.Tensor) -> float:
    return float(labels.float().mean()) if len(labels) else 0.0


def main() -> None:
    configure_logging()
    log = get_logger("safelens.scripts.train_multimodal_fusion")

    config = load_fusion_config()
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
        "loaded (fusion, text+image both permitted): train=%d dev=%d test=%d",
        len(train),
        len(dev),
        len(test),
    )

    image_processor, image_model = build_clip_encoder(config)
    text_encoder_adapter = TextArabicConfig(
        model_name="adapter",
        hf_model_name=config.text_model_name,
        hf_model_revision=config.text_model_revision,
    )
    text_tokenizer, text_model = build_text_encoder(text_encoder_adapter)
    log.info(
        "frozen encoders: CLIP=%s@%s AraBERT=%s@%s",
        config.clip_model_name,
        config.clip_revision,
        config.text_model_name,
        config.text_model_revision,
    )

    log.info("extracting fused embeddings (train/dev/test)...")
    t0 = time.perf_counter()
    train_fused = extract_fused_embeddings(
        train,
        RAW_IMAGE_ROOT,
        image_processor,
        image_model,
        text_tokenizer,
        text_model,
        device,
        config.max_seq_length,
    )
    dev_fused = extract_fused_embeddings(
        dev,
        RAW_IMAGE_ROOT,
        image_processor,
        image_model,
        text_tokenizer,
        text_model,
        device,
        config.max_seq_length,
    )
    test_fused = extract_fused_embeddings(
        test,
        RAW_IMAGE_ROOT,
        image_processor,
        image_model,
        text_tokenizer,
        text_model,
        device,
        config.max_seq_length,
    )
    embedding_extraction_time = time.perf_counter() - t0
    log.info("embedding extraction time: %.1fs", embedding_extraction_time)

    embed_dim = train_fused.fused_embeddings.shape[1]
    t0 = time.perf_counter()
    result = train_head(
        train_fused.fused_embeddings,
        train_fused.labels,
        dev_fused.fused_embeddings,
        dev_fused.labels,
        config.training,
        config.head,
        embed_dim=embed_dim,
    )
    training_time = time.perf_counter() - t0
    log.info(
        "training time: %.2fs, best_epoch=%d, epochs_run=%d, class_weights=%s",
        training_time,
        result.best_epoch,
        result.epochs_run,
        result.class_weights,
    )

    head = ClassificationHead(config.head, embed_dim=embed_dim)
    head.load_state_dict(result.best_state_dict)
    head.eval()

    with torch.no_grad():
        dev_probs = torch.softmax(head(dev_fused.fused_embeddings), dim=-1)[:, 1].numpy()
    dev_metrics_at_05 = compute_metrics(dev_fused.labels.tolist(), dev_probs, 0.5)
    sweep = sweep_thresholds(
        dev_fused.labels.tolist(), dev_probs.tolist(), config.decision_threshold.candidates
    )
    selected_threshold = select_threshold(sweep, config.decision_threshold.selection_metric)
    log.info("selected decision threshold (dev-only, secondary): %.2f", selected_threshold)

    # Test set: fixed 0.5 (primary) + selected threshold (secondary), single predict pass.
    with torch.no_grad():
        test_probs = torch.softmax(head(test_fused.fused_embeddings), dim=-1)[:, 1].numpy()
    test_metrics_at_05 = compute_metrics(test_fused.labels.tolist(), test_probs, 0.5)
    test_metrics_at_selected = compute_metrics(
        test_fused.labels.tolist(), test_probs, selected_threshold
    )
    test_pred_at_selected = (test_probs >= selected_threshold).astype(int)

    # --- Ablation: fusion model, one modality replaced by train-mean vector ---
    means = compute_modality_means(train_fused.image_embeddings, train_fused.text_embeddings)
    with torch.no_grad():
        test_image_only_input = build_image_only_input(test_fused.image_embeddings, means)
        test_text_only_input = build_text_only_input(test_fused.text_embeddings, means)
        fusion_image_only_probs = torch.softmax(head(test_image_only_input), dim=-1)[:, 1].numpy()
        fusion_text_only_probs = torch.softmax(head(test_text_only_input), dim=-1)[:, 1].numpy()
    fusion_image_only_metrics = compute_metrics(
        test_fused.labels.tolist(), fusion_image_only_probs, 0.5
    )
    fusion_text_only_metrics = compute_metrics(
        test_fused.labels.tolist(), fusion_text_only_probs, 0.5
    )
    log.info(
        "ablation @0.5: fusion(full) f1=%.4f | fusion(image-only-input) f1=%.4f | "
        "fusion(text-only-input) f1=%.4f",
        test_metrics_at_05["f1"],
        fusion_image_only_metrics["f1"],
        fusion_text_only_metrics["f1"],
    )

    # --- Complementarity: standalone Phase 5A/5B heads (reloaded, not retrained) ---
    clip_head_state, clip_config = load_clip_artifacts("v1")
    clip_standalone_head = ClassificationHead(
        clip_config.head, embed_dim=test_fused.image_embeddings.shape[1]
    )
    clip_standalone_head.load_state_dict(clip_head_state)
    clip_standalone_head.eval()

    arabert_head_state, arabert_config = load_text_artifacts("arabert", "v1")
    arabert_standalone_head = ClassificationHead(
        arabert_config.head, embed_dim=test_fused.text_embeddings.shape[1]
    )
    arabert_standalone_head.load_state_dict(arabert_head_state)
    arabert_standalone_head.eval()

    with torch.no_grad():
        standalone_image_probs = torch.softmax(
            clip_standalone_head(test_fused.image_embeddings), dim=-1
        )[:, 1].numpy()
        standalone_text_probs = torch.softmax(
            arabert_standalone_head(test_fused.text_embeddings), dim=-1
        )[:, 1].numpy()

    complementarity = analyze_complementarity(
        test_fused.example_ids,
        test_fused.labels.tolist(),
        standalone_text_probs.tolist(),
        standalone_image_probs.tolist(),
        test_probs.tolist(),
        threshold=0.5,
    )
    log.info("complementarity counts: %s", complementarity["counts"])

    error_analysis = find_error_examples(
        test_fused.example_ids,
        [ex.text for ex in test],
        [ex.image_path for ex in test],
        test_fused.labels.tolist(),
        test_probs.tolist(),
        0.5,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_precision_recall(
        dev_fused.labels.tolist(), dev_probs.tolist(), RESULTS_DIR / "precision_recall_curve.png"
    )
    plot_confusion_matrix(
        test_fused.labels.tolist(),
        test_pred_at_selected.tolist(),
        selected_threshold,
        RESULTS_DIR / "confusion_matrix.png",
    )

    # --- Latency: image-only stages, text-only stages, fusion head, multimodal end-to-end ---
    sample_example = test[0]
    sample_image = (
        __import__("PIL.Image", fromlist=["Image"])
        .open(RAW_IMAGE_ROOT / sample_example.image_path)
        .convert("RGB")
    )
    sample_pixel_values = image_processor(images=sample_image, return_tensors="pt")["pixel_values"]
    sample_text_encoding = text_tokenizer(
        sample_example.text,
        truncation=True,
        padding="max_length",
        max_length=config.max_seq_length,
        return_tensors="pt",
    )

    image_preprocessing_bench = benchmark_preprocessing_latency(image_processor, sample_image)
    clip_encoder_bench = benchmark_encoder_latency(image_model, sample_pixel_values, device)
    tokenization_bench = benchmark_tokenization_latency(
        text_tokenizer, sample_example.text, config.max_seq_length
    )
    text_encoder_bench = benchmark_text_encoder_latency(
        text_model,
        sample_text_encoding["input_ids"],
        sample_text_encoding["attention_mask"],
        device,
    )
    fusion_head_bench = benchmark_fusion_head_latency(head, test_fused.fused_embeddings[0:1])
    fusion_end_to_end_bench = benchmark_fusion_end_to_end_latency(
        image_processor,
        image_model,
        text_tokenizer,
        text_model,
        head,
        RAW_IMAGE_ROOT,
        sample_example.image_path,
        sample_example.text,
        config.max_seq_length,
        device,
    )

    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}

    experiment = {
        "model_name": config.model_name,
        "model_version": config.model_version,
        "clip_model_name": config.clip_model_name,
        "clip_revision": config.clip_revision,
        "text_model_name": config.text_model_name,
        "text_model_revision": config.text_model_revision,
        "config": config.model_dump(),
        "dataset_manifest_reference": str(MANIFEST_PATH),
        "dataset_manifest": manifest,
        "split_sizes": {"train": len(train), "dev": len(dev), "test": len(test)},
        "positive_rate": {
            "train": _pos_rate(train_fused.labels),
            "dev": _pos_rate(dev_fused.labels),
            "test": _pos_rate(test_fused.labels),
        },
        "missing_modality_representation": (
            "Train-mean embedding vector per modality, NOT a zero vector -- documented in "
            "safelens/models/multimodal/missing_modality.py. Not a learned gating mechanism."
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
            "fused_embed_dim": embed_dim,
        },
        "dev_metrics_at_threshold_0.5": dev_metrics_at_05,
        "threshold_sweep_dev_only_secondary": sweep,
        "selected_threshold_secondary": selected_threshold,
        "test_metrics_at_threshold_0.5_PRIMARY": test_metrics_at_05,
        "test_metrics_at_selected_threshold_secondary": test_metrics_at_selected,
        "ablation": {
            "fusion_full_test_metrics_at_0.5": test_metrics_at_05,
            "fusion_image_only_input_test_metrics_at_0.5": fusion_image_only_metrics,
            "fusion_text_only_input_test_metrics_at_0.5": fusion_text_only_metrics,
        },
        "standalone_vs_fusion_ablation_note": (
            "Standalone CLIP/AraBERT (Phase 5A/5B, own trained heads) are DIFFERENT models "
            "from Fusion(image-only-input)/Fusion(text-only-input) (this fusion head, one "
            "input replaced by a train-mean placeholder). Both are reported -- see "
            "complementarity_analysis and required_comparison in docs/evaluation.md."
        ),
        "complementarity_analysis": complementarity,
        "error_analysis": error_analysis,
        "inference_benchmark": {
            "environment": f"Measured on {platform.platform()}, device={device}",
            "image_preprocessing": image_preprocessing_bench.__dict__,
            "clip_encoder": clip_encoder_bench.__dict__,
            "text_tokenization": tokenization_bench.__dict__,
            "text_encoder": text_encoder_bench.__dict__,
            "fusion_head": fusion_head_bench.__dict__,
            "multimodal_end_to_end": fusion_end_to_end_bench.__dict__,
        },
        "environment": environment_metadata(),
    }
    (RESULTS_DIR / "experiment.json").write_text(
        json.dumps(experiment, indent=2, sort_keys=True, ensure_ascii=False)
    )
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
        "TEST @ 0.5 (PRIMARY): f1=%.4f pr_auc=%.4f recall=%.4f fnr=%.4f",
        test_metrics_at_05["f1"],
        test_metrics_at_05["pr_auc"],
        test_metrics_at_05["recall"],
        test_metrics_at_05["false_negative_rate"],
    )


if __name__ == "__main__":
    main()
