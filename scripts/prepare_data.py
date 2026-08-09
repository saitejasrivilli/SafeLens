#!/usr/bin/env python3
"""make data-prepare -- validate, dedup, split, leakage-check, and write the
versioned processed dataset + manifest.

Pipeline order (leakage-safe by construction):
  raw rows -> schema validation (drop malformed/invalid)
           -> exact dedup (id + content)
           -> stratified split (seed=42)
           -> leakage check across the resulting splits
           -> write processed/*.jsonl + manifest.json
"""

from __future__ import annotations

import json
from pathlib import Path

from safelens.data.manifest import build_manifest, write_manifest
from safelens.data.mapping import MalformedRowError, ValidationError, civil_comments_row_to_example
from safelens.data.preprocessing.dedup import deduplicate
from safelens.data.preprocessing.leakage import check_leakage
from safelens.data.preprocessing.split import SplitConfig, split_dataset
from safelens.data.schema import ModerationExample
from safelens.utils.logging import configure_logging, get_logger

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "civil_comments" / "pool.jsonl"
METADATA_PATH = ROOT / "data" / "raw" / "civil_comments" / "metadata.json"
PROCESSED_DIR = ROOT / "data" / "processed" / "civil_comments"
MANIFEST_PATH = ROOT / "data" / "manifests" / "civil_comments_manifest.json"


def main() -> None:
    configure_logging()
    log = get_logger("safelens.scripts.prepare_data")

    if not RAW_PATH.exists():
        log.error("%s not found. Run `make data-download` first.", RAW_PATH)
        raise SystemExit(1)

    metadata = json.loads(METADATA_PATH.read_text())
    dataset_version = (
        f"{metadata['dataset_name']}@{metadata['dataset_split']}:0-{metadata['actual_rows']}"
    )

    rows = [json.loads(line) for line in RAW_PATH.read_text().splitlines() if line.strip()]

    valid: list[ModerationExample] = []
    for row in rows:
        try:
            valid.append(civil_comments_row_to_example(row, dataset_version=dataset_version))
        except (MalformedRowError, ValidationError):
            continue
    log.info("schema-valid rows: %d / %d", len(valid), len(rows))

    dedup_result = deduplicate(valid)
    log.info(
        "after dedup: %d unique (removed %d dup-id, %d dup-content, %d near-dup detected)",
        len(dedup_result.unique),
        len(dedup_result.duplicate_ids_removed),
        len(dedup_result.duplicate_content_removed),
        dedup_result.normalized_duplicate_examples,
    )

    splits = split_dataset(dedup_result.unique, SplitConfig(seed=42))
    for name, exs in splits.items():
        log.info("split %s: %d examples", name, len(exs))

    leakage = check_leakage(splits)
    if not leakage.is_clean:
        log.error("leakage detected: %s", leakage)
        raise SystemExit(1)
    log.info("leakage check: clean (0 id overlaps, 0 normalized-text overlaps)")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, exs in splits.items():
        out_path = PROCESSED_DIR / f"{name}.jsonl"
        with out_path.open("w") as f:
            for ex in exs:
                f.write(ex.model_dump_json() + "\n")
        log.info("wrote %s (%d rows)", out_path, len(exs))

    manifest = build_manifest(
        dataset_name=metadata["dataset_name"],
        dataset_version=dataset_version,
        source_url=metadata["source_url"],
        license_id=metadata["license"],
        retrieval_date=metadata["retrieval_date"],
        raw_path=RAW_PATH,
        splits=splits,
        seed=42,
    )
    write_manifest(manifest, MANIFEST_PATH)
    log.info("manifest written to %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()
