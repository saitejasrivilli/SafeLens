"""Machine-readable manifest for the Prop2Hate-Meme ingestion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PREPROCESSING_VERSION = "v2"  # v2: adds train-side exact-caption leakage remediation


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_distribution_shift_stats(
    *,
    original_train_positive_rate: float,
    clean_train_positive_rate: float,
    dev_positive_rate: float,
    test_positive_rate: float,
) -> dict[str, Any]:
    """No explanation is asserted for the shift -- the official dataset
    documentation (checked directly) states no split-construction rationale
    for the prevalence difference, so this only records the measured
    numbers, not a cause."""
    return {
        "train_positive_rate": original_train_positive_rate,
        "clean_train_positive_rate": clean_train_positive_rate,
        "dev_positive_rate": dev_positive_rate,
        "test_positive_rate": test_positive_rate,
        "absolute_difference_test_minus_train": test_positive_rate - original_train_positive_rate,
        "relative_ratio_test_over_train": (
            test_positive_rate / original_train_positive_rate
            if original_train_positive_rate
            else float("inf")
        ),
        "authoritative_explanation_found": False,
        "note": (
            "No rationale for the train/dev vs. test prevalence difference was found in the "
            "official QCRI/Prop2Hate-Meme dataset card or README. Documented as an observed "
            "distribution shift, not an inferred cause."
        ),
    }


def build_manifest(
    *,
    ingestion_metadata: dict[str, Any],
    validation_report: dict[str, Any],
    dedup_report: dict[str, Any],
    leakage_report: dict[str, Any],
    processed_dir: Path,
    remediation_summary: dict[str, Any] | None = None,
    distribution_shift: dict[str, Any] | None = None,
) -> dict[str, Any]:
    jsonl_hashes = {
        split: sha256_file(processed_dir / f"{split}.jsonl")
        for split in ("train", "dev", "test")
        if (processed_dir / f"{split}.jsonl").exists()
    }

    manifest: dict[str, Any] = {
        "dataset_name": ingestion_metadata["dataset_name"],
        "dataset_revision": ingestion_metadata["dataset_revision"],
        "source_url": ingestion_metadata["source_url"],
        "license": ingestion_metadata["license"],
        "attribution": ingestion_metadata["attribution"],
        "retrieval_date": ingestion_metadata["retrieval_date"],
        "split_counts": ingestion_metadata["split_counts"],
        "expected_split_counts": ingestion_metadata["expected_split_counts"],
        "processed_jsonl_hashes": jsonl_hashes,
        "validation_summary": {
            "image_validation": validation_report["image_validation_summary"],
            "arabic_text_fraction_overall": validation_report["arabic_text_fraction_overall"],
            "label_distribution_by_split": {
                split: data.get("label_distribution", {})
                for split, data in validation_report["splits"].items()
                if "label_distribution" in data
            },
        },
        "duplicate_summary": {
            "duplicate_ids": len(dedup_report["duplicate_ids"]),
            "duplicate_texts": len(dedup_report["duplicate_texts"]),
            "duplicate_image_hashes": len(dedup_report["duplicate_image_hashes"]),
            "duplicate_pairs": len(dedup_report["duplicate_pairs"]),
        },
        "leakage_summary": {
            "is_clean": leakage_report["is_clean"],
            "id_overlaps": {k: len(v) for k, v in leakage_report["id_overlaps"].items()},
            "text_overlaps": {k: len(v) for k, v in leakage_report["text_overlaps"].items()},
            "image_hash_overlaps": {
                k: len(v) for k, v in leakage_report["image_hash_overlaps"].items()
            },
            "pair_overlaps": {k: len(v) for k, v in leakage_report["pair_overlaps"].items()},
        },
        "preprocessing_version": PREPROCESSING_VERSION,
    }

    if remediation_summary is not None:
        manifest["leakage_remediation"] = remediation_summary
    if distribution_shift is not None:
        manifest["distribution_shift"] = distribution_shift

    return manifest


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
