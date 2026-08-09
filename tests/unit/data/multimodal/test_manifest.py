import json
from pathlib import Path

import pytest

from safelens.data.multimodal.manifest import (
    build_distribution_shift_stats,
    build_manifest,
    write_manifest,
)

INGESTION_METADATA = {
    "dataset_name": "QCRI/Prop2Hate-Meme",
    "dataset_revision": "test-revision",
    "source_url": "https://huggingface.co/datasets/QCRI/Prop2Hate-Meme",
    "license": "CC-BY-NC-SA-4.0",
    "attribution": "QCRI",
    "retrieval_date": "2026-01-01T00:00:00+00:00",
    "split_counts": {"train": 2, "dev": 1, "test": 1},
    "expected_split_counts": {"train": 2143, "dev": 312, "test": 606},
}

VALIDATION_REPORT = {
    "image_validation_summary": {"total_checked": 4, "valid": 4, "corrupted": 0, "missing": 0},
    "arabic_text_fraction_overall": 1.0,
    "splits": {
        "train": {
            "label_distribution": {
                "hate_label": {
                    "counts": {"hateful": 1, "not-hateful": 1},
                    "total": 2,
                    "positive_rate": 0.5,
                }
            }
        },
    },
}

DEDUP_REPORT = {
    "duplicate_ids": [],
    "duplicate_texts": [],
    "duplicate_image_hashes": [],
    "duplicate_pairs": [],
}
LEAKAGE_REPORT = {
    "is_clean": True,
    "id_overlaps": {},
    "text_overlaps": {},
    "image_hash_overlaps": {},
    "pair_overlaps": {},
}


def test_build_manifest_fields(tmp_path: Path):
    for split in ("train", "dev", "test"):
        (tmp_path / f"{split}.jsonl").write_text('{"example_id": "x"}\n')

    manifest = build_manifest(
        ingestion_metadata=INGESTION_METADATA,
        validation_report=VALIDATION_REPORT,
        dedup_report=DEDUP_REPORT,
        leakage_report=LEAKAGE_REPORT,
        processed_dir=tmp_path,
    )

    for key in [
        "dataset_name",
        "dataset_revision",
        "source_url",
        "license",
        "attribution",
        "retrieval_date",
        "split_counts",
        "expected_split_counts",
        "processed_jsonl_hashes",
        "validation_summary",
        "duplicate_summary",
        "leakage_summary",
        "preprocessing_version",
    ]:
        assert key in manifest

    assert manifest["license"] == "CC-BY-NC-SA-4.0"
    assert set(manifest["processed_jsonl_hashes"].keys()) == {"train", "dev", "test"}
    assert manifest["leakage_summary"]["is_clean"] is True


def test_write_manifest_round_trip(tmp_path: Path):
    for split in ("train", "dev", "test"):
        (tmp_path / f"{split}.jsonl").write_text('{"example_id": "x"}\n')

    manifest = build_manifest(
        ingestion_metadata=INGESTION_METADATA,
        validation_report=VALIDATION_REPORT,
        dedup_report=DEDUP_REPORT,
        leakage_report=LEAKAGE_REPORT,
        processed_dir=tmp_path,
    )
    out_path = tmp_path / "manifest.json"
    write_manifest(manifest, out_path)
    assert json.loads(out_path.read_text()) == manifest


def test_manifest_includes_remediation_and_distribution_shift_when_provided(tmp_path: Path):
    for split in ("train", "dev", "test"):
        (tmp_path / f"{split}.jsonl").write_text('{"example_id": "x"}\n')

    remediation_summary = {
        "rule": "exact text match",
        "original_train_count": 2143,
        "removed_train_count": 2,
        "final_train_count": 2141,
        "dev_count": 312,
        "test_count": 606,
        "removed_examples": [{"example_id": "a", "text": "leaked"}],
        "post_remediation_leakage_clean": True,
    }
    distribution_shift = build_distribution_shift_stats(
        original_train_positive_rate=0.0994,
        clean_train_positive_rate=0.0994,
        dev_positive_rate=0.0994,
        test_positive_rate=0.2541,
    )

    manifest = build_manifest(
        ingestion_metadata=INGESTION_METADATA,
        validation_report=VALIDATION_REPORT,
        dedup_report=DEDUP_REPORT,
        leakage_report=LEAKAGE_REPORT,
        processed_dir=tmp_path,
        remediation_summary=remediation_summary,
        distribution_shift=distribution_shift,
    )

    assert manifest["leakage_remediation"]["removed_train_count"] == 2
    assert manifest["leakage_remediation"]["final_train_count"] == 2141
    assert manifest["distribution_shift"]["test_positive_rate"] == 0.2541
    assert manifest["distribution_shift"]["authoritative_explanation_found"] is False


def test_distribution_shift_stats_calculation():
    stats = build_distribution_shift_stats(
        original_train_positive_rate=0.10,
        clean_train_positive_rate=0.10,
        dev_positive_rate=0.10,
        test_positive_rate=0.25,
    )
    assert stats["absolute_difference_test_minus_train"] == pytest.approx(0.15)
    assert stats["relative_ratio_test_over_train"] == pytest.approx(2.5)
