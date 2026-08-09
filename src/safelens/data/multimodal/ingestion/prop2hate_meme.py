"""Ingestion for QCRI/Prop2Hate-Meme (CC-BY-NC-SA-4.0).

Source: https://huggingface.co/datasets/QCRI/Prop2Hate-Meme
License: CC-BY-NC-SA-4.0 -- non-commercial, share-alike, attribution required.
Verified directly from the dataset repo's own README/cardData, not inferred.

This is a SEPARATE experimental track from Phase 2-4's civil_comments
pipeline. Never redistributed: this module only extracts the dataset into
`data/multimodal/` (entirely gitignored except manifests) for local use.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DATASET_NAME = "QCRI/Prop2Hate-Meme"
DATASET_LICENSE = "CC-BY-NC-SA-4.0"
SOURCE_URL = "https://huggingface.co/datasets/QCRI/Prop2Hate-Meme"
ATTRIBUTION = (
    "Prop2Hate-Meme dataset by QCRI (Qatar Computing Research Institute), "
    "extending the ArMeme corpus with hate-speech annotations. "
    "See https://arxiv.org/abs/2409.07246."
)
EXPECTED_SPLIT_SIZES = {"train": 2143, "dev": 312, "test": 606}


class DatasetAlreadyExistsError(RuntimeError):
    pass


@dataclass(frozen=True)
class IngestionResult:
    processed_dir: Path
    metadata_path: Path
    split_counts: dict[str, int]


def _image_filename(example_id: str) -> str:
    """Deterministic, filesystem-safe filename -- the dataset's own `id`
    field is a full nested path string, not safe to use as a filename."""
    return hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:16] + ".jpg"


def _dataset_revision() -> str:
    """Best-effort: query the HF Hub API for the exact commit sha. Falls
    back to "unknown" if offline -- never blocks ingestion on this."""
    try:
        import requests

        resp = requests.get(f"https://huggingface.co/api/datasets/{DATASET_NAME}", timeout=10)
        resp.raise_for_status()
        return str(resp.json().get("sha", "unknown"))
    except Exception:
        return "unknown"


def download_prop2hate_meme(output_dir: Path, force: bool = False) -> IngestionResult:
    """Loads the official train/dev/test splits via the `datasets` library
    (pinned by revision where possible), preserving the official split
    assignment exactly -- no reshuffling, no resplitting. Extracts each
    image to disk and writes one JSONL file per split.
    """
    from datasets import load_dataset

    processed_dir = output_dir / "prop2hate_meme"
    metadata_path = processed_dir / "metadata.json"

    if metadata_path.exists() and not force:
        raise DatasetAlreadyExistsError(
            f"{metadata_path} already exists. Use force=True to re-download."
        )

    revision = _dataset_revision()
    ds = load_dataset(DATASET_NAME)

    split_counts: dict[str, int] = {}
    for split_name in ("train", "dev", "test"):
        split = ds[split_name]
        images_dir = processed_dir / "images" / split_name
        images_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = processed_dir / f"{split_name}.jsonl"
        with jsonl_path.open("w") as f:
            for row in split:
                image_filename = _image_filename(row["id"])
                image_path = images_dir / image_filename
                row["image"].convert("RGB").save(image_path, format="JPEG")

                record: dict[str, Any] = {
                    "example_id": row["id"],
                    "text": row["text"],
                    "image_path": str(image_path.relative_to(processed_dir)),
                    "source_img_path": row["img_path"],
                    "hate_label": row["hate_label"],
                    "prop_label": row["prop_label"],
                    "hate_fine_grained_label": row["hate_fine_grained_label"],
                    "split": split_name,
                    "dataset_version": f"{DATASET_NAME}@{revision}",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        split_counts[split_name] = len(split)

    metadata = {
        "dataset_name": DATASET_NAME,
        "dataset_revision": revision,
        "license": DATASET_LICENSE,
        "source_url": SOURCE_URL,
        "attribution": ATTRIBUTION,
        "retrieval_date": datetime.now(UTC).isoformat(),
        "split_counts": split_counts,
        "expected_split_counts": EXPECTED_SPLIT_SIZES,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    return IngestionResult(
        processed_dir=processed_dir, metadata_path=metadata_path, split_counts=split_counts
    )
