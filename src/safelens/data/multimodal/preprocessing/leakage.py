"""Cross-split leakage checks: train<->dev, train<->test, dev<->test.

Report-only. The official split is never modified based on these findings
-- per instructions, a serious leakage problem found here must be reported
and escalated, not silently patched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from safelens.data.multimodal.preprocessing.hashing import image_sha256
from safelens.data.multimodal.schema import MultimodalExample


@dataclass(frozen=True)
class MultimodalLeakageReport:
    id_overlaps: dict[str, list[str]] = field(default_factory=dict)
    text_overlaps: dict[str, list[str]] = field(default_factory=dict)
    image_hash_overlaps: dict[str, list[str]] = field(default_factory=dict)
    pair_overlaps: dict[str, list[str]] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        return not (
            self.id_overlaps or self.text_overlaps or self.image_hash_overlaps or self.pair_overlaps
        )


def check_leakage(
    splits: dict[str, list[MultimodalExample]], processed_dir: Path
) -> MultimodalLeakageReport:
    ids_by_split = {name: {ex.example_id for ex in exs} for name, exs in splits.items()}
    texts_by_split = {name: {ex.text for ex in exs} for name, exs in splits.items()}
    hashes_by_split = {
        name: {image_sha256(processed_dir / ex.image_path) for ex in exs}
        for name, exs in splits.items()
    }
    pairs_by_split = {
        name: {f"{ex.text}|||{image_sha256(processed_dir / ex.image_path)}" for ex in exs}
        for name, exs in splits.items()
    }

    id_overlaps: dict[str, list[str]] = {}
    text_overlaps: dict[str, list[str]] = {}
    hash_overlaps: dict[str, list[str]] = {}
    pair_overlaps: dict[str, list[str]] = {}

    for a, b in combinations(splits.keys(), 2):
        key = f"{a}<->{b}"
        if shared := ids_by_split[a] & ids_by_split[b]:
            id_overlaps[key] = sorted(shared)
        if shared := texts_by_split[a] & texts_by_split[b]:
            text_overlaps[key] = sorted(shared)
        if shared := hashes_by_split[a] & hashes_by_split[b]:
            hash_overlaps[key] = sorted(shared)
        if shared := pairs_by_split[a] & pairs_by_split[b]:
            pair_overlaps[key] = sorted(shared)

    return MultimodalLeakageReport(
        id_overlaps=id_overlaps,
        text_overlaps=text_overlaps,
        image_hash_overlaps=hash_overlaps,
        pair_overlaps=pair_overlaps,
    )
