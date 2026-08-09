#!/usr/bin/env python3
"""make multimodal-validate -- validate the ingested Prop2Hate-Meme data:
schema/image/text checks, duplicate detection, cross-split leakage checks,
and manifest generation. Never modifies the official split."""

from __future__ import annotations

import json
from pathlib import Path

from safelens.data.multimodal.manifest import build_manifest, write_manifest
from safelens.data.multimodal.preprocessing.dedup import find_duplicates
from safelens.data.multimodal.preprocessing.leakage import check_leakage
from safelens.data.multimodal.validation.report import build_validation_report, load_valid_examples
from safelens.utils.logging import configure_logging, get_logger

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "multimodal" / "raw" / "prop2hate_meme"
METADATA_PATH = PROCESSED_DIR / "metadata.json"
REPORT_PATH = ROOT / "benchmarks" / "results" / "multimodal" / "data_validation_report.json"
MANIFEST_PATH = ROOT / "data" / "multimodal" / "manifests" / "prop2hate_meme_manifest.json"


def main() -> None:
    configure_logging()
    log = get_logger("safelens.scripts.validate_multimodal_data")

    if not METADATA_PATH.exists():
        log.error("%s not found. Run `make multimodal-download` first.", METADATA_PATH)
        raise SystemExit(1)

    ingestion_metadata = json.loads(METADATA_PATH.read_text())

    log.info("building validation report (schema, image, text checks)...")
    validation_report = build_validation_report(PROCESSED_DIR)
    for split_name, data in validation_report["splits"].items():
        log.info(
            "split=%s total=%d valid=%d malformed=%d text_issues=%d image_issues=%d",
            split_name,
            data["total_rows"],
            data["valid_rows"],
            data["malformed_rows"],
            data["text_issues"],
            data["image_issues"],
        )
        if data["malformed_rows"] or data["image_issues"]:
            log.warning(
                "split=%s has %d malformed rows and %d image issues -- see report for detail",
                split_name,
                data["malformed_rows"],
                data["image_issues"],
            )

    log.info("actual expected split sizes vs. documented expectation:")
    log.info(
        "actual=%s expected=%s",
        ingestion_metadata["split_counts"],
        ingestion_metadata["expected_split_counts"],
    )

    examples_by_split = load_valid_examples(PROCESSED_DIR)
    all_examples = [ex for exs in examples_by_split.values() for ex in exs]

    log.info("checking duplicates (id, text, image hash, text+image pair)...")
    dedup_report = find_duplicates(all_examples, PROCESSED_DIR)
    log.info(
        "duplicates: ids=%d texts=%d image_hashes=%d pairs=%d",
        len(dedup_report.duplicate_ids),
        len(dedup_report.duplicate_texts),
        len(dedup_report.duplicate_image_hashes),
        len(dedup_report.duplicate_pairs),
    )

    log.info("checking cross-split leakage (train<->dev, train<->test, dev<->test)...")
    leakage_report = check_leakage(examples_by_split, PROCESSED_DIR)
    if not leakage_report.is_clean:
        log.error("LEAKAGE DETECTED: %s", leakage_report)
    else:
        log.info("leakage check: clean")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(validation_report, indent=2, sort_keys=True, ensure_ascii=False)
    )
    log.info("validation report written to %s", REPORT_PATH)

    manifest = build_manifest(
        ingestion_metadata=ingestion_metadata,
        validation_report=validation_report,
        dedup_report={
            "duplicate_ids": dedup_report.duplicate_ids,
            "duplicate_texts": dedup_report.duplicate_texts,
            "duplicate_image_hashes": dedup_report.duplicate_image_hashes,
            "duplicate_pairs": dedup_report.duplicate_pairs,
        },
        leakage_report={
            "is_clean": leakage_report.is_clean,
            "id_overlaps": leakage_report.id_overlaps,
            "text_overlaps": leakage_report.text_overlaps,
            "image_hash_overlaps": leakage_report.image_hash_overlaps,
            "pair_overlaps": leakage_report.pair_overlaps,
        },
        processed_dir=PROCESSED_DIR,
    )
    write_manifest(manifest, MANIFEST_PATH)
    log.info("manifest written to %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()
