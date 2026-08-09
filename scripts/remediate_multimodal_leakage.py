#!/usr/bin/env python3
"""make multimodal-remediate -- produces a leakage-clean PROCESSED training
split for Prop2Hate-Meme.

The official RAW train/dev/test split (data/multimodal/raw/) is never
modified. This script only writes a filtered copy of train under
data/multimodal/processed/, dropping any training example whose caption
text exactly matches a dev or test caption. Dev and test are copied
unchanged (same rows, same image references) so downstream training code
has one consistent directory to read from.
"""

from __future__ import annotations

import json
from pathlib import Path

from safelens.data.multimodal.manifest import (
    build_distribution_shift_stats,
    build_manifest,
    write_manifest,
)
from safelens.data.multimodal.preprocessing.dedup import find_duplicates
from safelens.data.multimodal.preprocessing.leakage import check_leakage
from safelens.data.multimodal.preprocessing.remediation import remediate_train_leakage
from safelens.data.multimodal.schema import MultimodalExample
from safelens.data.multimodal.validation.report import build_validation_report, load_valid_examples
from safelens.utils.logging import configure_logging, get_logger

ROOT = Path(__file__).resolve().parents[1]
RAW_PROCESSED_DIR = ROOT / "data" / "multimodal" / "raw" / "prop2hate_meme"
METADATA_PATH = RAW_PROCESSED_DIR / "metadata.json"
PROCESSED_DIR = ROOT / "data" / "multimodal" / "processed" / "prop2hate_meme"
REMEDIATION_REPORT_PATH = (
    ROOT / "benchmarks" / "results" / "multimodal" / "leakage_remediation_report.json"
)
MANIFEST_PATH = ROOT / "data" / "multimodal" / "manifests" / "prop2hate_meme_manifest.json"


def _positive_rate(examples: list[MultimodalExample]) -> float:
    if not examples:
        return 0.0
    return sum(ex.hate_label for ex in examples) / len(examples)


def _write_jsonl(path: Path, examples: list[MultimodalExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")


def main() -> None:
    configure_logging()
    log = get_logger("safelens.scripts.remediate_multimodal_leakage")

    if not METADATA_PATH.exists():
        log.error("%s not found. Run `make multimodal-download` first.", METADATA_PATH)
        raise SystemExit(1)

    examples_by_split = load_valid_examples(RAW_PROCESSED_DIR)
    train, dev, test = (
        examples_by_split["train"],
        examples_by_split["dev"],
        examples_by_split["test"],
    )
    log.info("loaded (raw, unmodified): train=%d dev=%d test=%d", len(train), len(dev), len(test))

    result = remediate_train_leakage(train, dev, test)
    log.info(
        "remediation: original_train=%d removed=%d final_train=%d",
        result.original_count,
        result.removed_count,
        result.final_count,
    )
    for ex in result.removed_examples:
        log.info("removed train example (leaked caption): id=%s text=%r", ex.example_id, ex.text)

    # Write the processed, leakage-clean split. Dev/test copied unchanged.
    _write_jsonl(PROCESSED_DIR / "train.jsonl", result.clean_train)
    _write_jsonl(PROCESSED_DIR / "dev.jsonl", dev)
    _write_jsonl(PROCESSED_DIR / "test.jsonl", test)
    log.info("processed leakage-clean split written to %s", PROCESSED_DIR)

    # Re-verify: clean train vs. dev/test must now have zero text overlap.
    post_splits = {"train": result.clean_train, "dev": dev, "test": test}
    post_leakage = check_leakage(post_splits, RAW_PROCESSED_DIR)
    log.info("post-remediation leakage check: is_clean=%s", post_leakage.is_clean)
    if not post_leakage.is_clean:
        log.error("REMEDIATION DID NOT FULLY RESOLVE LEAKAGE: %s", post_leakage)

    all_post = result.clean_train + dev + test
    post_dedup = find_duplicates(all_post, RAW_PROCESSED_DIR)

    distribution_shift = build_distribution_shift_stats(
        original_train_positive_rate=_positive_rate(train),
        clean_train_positive_rate=_positive_rate(result.clean_train),
        dev_positive_rate=_positive_rate(dev),
        test_positive_rate=_positive_rate(test),
    )
    log.info("distribution shift stats: %s", distribution_shift)

    remediation_summary = {
        "rule": (
            "Exact-text match only (no normalization, no semantic dedup): a training example "
            "is removed from the processed training split if its caption text exactly matches "
            "any dev or test caption text. Dev and test are never modified. The official raw "
            "split (data/multimodal/raw/) is never modified -- this is a processed-split-only "
            "remediation."
        ),
        "original_train_count": result.original_count,
        "removed_train_count": result.removed_count,
        "final_train_count": result.final_count,
        "dev_count": len(dev),
        "test_count": len(test),
        "removed_examples": [
            {"example_id": ex.example_id, "text": ex.text} for ex in result.removed_examples
        ],
        "post_remediation_leakage_clean": post_leakage.is_clean,
        "post_remediation_duplicate_summary": {
            "duplicate_ids": len(post_dedup.duplicate_ids),
            "duplicate_texts": len(post_dedup.duplicate_texts),
            "duplicate_image_hashes": len(post_dedup.duplicate_image_hashes),
            "duplicate_pairs": len(post_dedup.duplicate_pairs),
        },
    }

    REMEDIATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REMEDIATION_REPORT_PATH.write_text(
        json.dumps(
            {
                "remediation": remediation_summary,
                "distribution_shift": distribution_shift,
                "post_remediation_leakage": {
                    "is_clean": post_leakage.is_clean,
                    "id_overlaps": post_leakage.id_overlaps,
                    "text_overlaps": post_leakage.text_overlaps,
                    "image_hash_overlaps": post_leakage.image_hash_overlaps,
                    "pair_overlaps": post_leakage.pair_overlaps,
                },
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    log.info("remediation report written to %s", REMEDIATION_REPORT_PATH)

    # Rebuild the manifest with the original ingestion/validation/dedup/leakage
    # context plus the new remediation + distribution-shift sections.
    ingestion_metadata = json.loads(METADATA_PATH.read_text())
    validation_report = build_validation_report(RAW_PROCESSED_DIR)
    raw_all_examples = train + dev + test
    raw_dedup = find_duplicates(raw_all_examples, RAW_PROCESSED_DIR)
    raw_leakage = check_leakage(examples_by_split, RAW_PROCESSED_DIR)

    manifest = build_manifest(
        ingestion_metadata=ingestion_metadata,
        validation_report=validation_report,
        dedup_report={
            "duplicate_ids": raw_dedup.duplicate_ids,
            "duplicate_texts": raw_dedup.duplicate_texts,
            "duplicate_image_hashes": raw_dedup.duplicate_image_hashes,
            "duplicate_pairs": raw_dedup.duplicate_pairs,
        },
        leakage_report={
            "is_clean": raw_leakage.is_clean,
            "id_overlaps": raw_leakage.id_overlaps,
            "text_overlaps": raw_leakage.text_overlaps,
            "image_hash_overlaps": raw_leakage.image_hash_overlaps,
            "pair_overlaps": raw_leakage.pair_overlaps,
        },
        processed_dir=RAW_PROCESSED_DIR,
        remediation_summary=remediation_summary,
        distribution_shift=distribution_shift,
    )
    write_manifest(manifest, MANIFEST_PATH)
    log.info("manifest updated at %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()
