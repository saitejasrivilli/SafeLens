"""Cross-split leakage checks. Must run after splitting, before splits are
treated as trustworthy for training/evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from safelens.data.preprocessing.normalize import normalize_text
from safelens.data.schema import ModerationExample


@dataclass(frozen=True)
class LeakageReport:
    id_overlaps: dict[str, list[str]] = field(default_factory=dict)
    normalized_text_overlaps: dict[str, list[str]] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        return not self.id_overlaps and not self.normalized_text_overlaps


def check_leakage(splits: dict[str, list[ModerationExample]]) -> LeakageReport:
    id_overlaps: dict[str, list[str]] = {}
    text_overlaps: dict[str, list[str]] = {}

    ids_by_split = {name: {ex.content_id for ex in exs} for name, exs in splits.items()}
    # Text that normalizes to "" (pure punctuation/emoji/symbols) carries no
    # comparable content, so it is excluded from the overlap check — treating
    # it as shared would flag unrelated examples as duplicates just because
    # both happened to strip down to nothing.
    norm_by_split = {
        name: {normalize_text(ex.text) for ex in exs} - {""} for name, exs in splits.items()
    }

    for a, b in combinations(splits.keys(), 2):
        shared_ids = ids_by_split[a] & ids_by_split[b]
        if shared_ids:
            id_overlaps[f"{a}<->{b}"] = sorted(shared_ids)

        shared_norm = norm_by_split[a] & norm_by_split[b]
        if shared_norm:
            text_overlaps[f"{a}<->{b}"] = sorted(shared_norm)

    return LeakageReport(id_overlaps=id_overlaps, normalized_text_overlaps=text_overlaps)
