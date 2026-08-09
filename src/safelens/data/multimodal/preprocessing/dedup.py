"""Duplicate detection (report-only -- never silently removes records).

Official train/dev/test splits are never modified by this module.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from safelens.data.multimodal.preprocessing.hashing import image_sha256
from safelens.data.multimodal.schema import MultimodalExample


@dataclass(frozen=True)
class DedupReport:
    duplicate_ids: list[str] = field(default_factory=list)
    duplicate_texts: list[str] = field(default_factory=list)
    duplicate_image_hashes: list[str] = field(default_factory=list)
    duplicate_pairs: list[str] = field(default_factory=list)  # "text|||image_hash" keys


def _duplicates(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(v for v, c in counts.items() if c > 1)


def find_duplicates(examples: list[MultimodalExample], processed_dir: Path) -> DedupReport:
    ids = [ex.example_id for ex in examples]
    texts = [ex.text for ex in examples]
    image_hashes = [image_sha256(processed_dir / ex.image_path) for ex in examples]
    pair_keys = [f"{t}|||{h}" for t, h in zip(texts, image_hashes, strict=True)]

    return DedupReport(
        duplicate_ids=_duplicates(ids),
        duplicate_texts=_duplicates(texts),
        duplicate_image_hashes=_duplicates(image_hashes),
        duplicate_pairs=_duplicates(pair_keys),
    )
