"""Machine-readable dataset manifest generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from safelens.data.schema import ModerationExample
from safelens.data.validation.report import LABEL_NAMES

PREPROCESSING_VERSION = "v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_examples(examples: list[ModerationExample]) -> str:
    """Order-independent content hash: sort by content_id first so the hash
    only changes when the actual example set changes, not list ordering."""
    payload = "\n".join(
        f"{ex.content_id}|{ex.text}"
        for ex in sorted(examples, key=lambda e: e.content_id)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(
    *,
    dataset_name: str,
    dataset_version: str,
    source_url: str,
    license_id: str,
    retrieval_date: str,
    raw_path: Path,
    splits: dict[str, list[ModerationExample]],
    seed: int,
) -> dict[str, Any]:
    all_examples = [ex for exs in splits.values() for ex in exs]

    label_distribution = {}
    for name in LABEL_NAMES:
        scores = [getattr(ex.labels, name) for ex in all_examples]
        label_distribution[name] = (
            sum(1 for s in scores if s >= 0.5) / len(scores) if scores else 0.0
        )

    return {
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "source_url": source_url,
        "license": license_id,
        "retrieval_date": retrieval_date,
        "raw_data_hash": sha256_file(raw_path),
        "processed_data_hash": sha256_examples(all_examples),
        "num_records": len(all_examples),
        "label_distribution_fraction_ge_0.5": label_distribution,
        "split_sizes": {name: len(exs) for name, exs in splits.items()},
        "preprocessing_version": PREPROCESSING_VERSION,
        "random_seed": seed,
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
