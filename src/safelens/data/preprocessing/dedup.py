"""Deterministic duplicate detection: exact ID, exact content, normalized text."""

from __future__ import annotations

from dataclasses import dataclass, field

from safelens.data.preprocessing.normalize import normalize_text
from safelens.data.schema import ModerationExample


@dataclass(frozen=True)
class DedupResult:
    unique: list[ModerationExample]
    duplicate_ids_removed: list[str] = field(default_factory=list)
    duplicate_content_removed: list[str] = field(default_factory=list)
    normalized_duplicate_groups: int = 0
    normalized_duplicate_examples: int = 0


def deduplicate(examples: list[ModerationExample]) -> DedupResult:
    """Remove exact duplicate IDs and exact duplicate text (keeping the first
    occurrence in input order, for determinism). Normalized-text duplicates
    are detected and counted, but not removed — near-identical comments can
    still be legitimately distinct moderation examples."""
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    unique: list[ModerationExample] = []
    dup_ids: list[str] = []
    dup_content: list[str] = []

    for ex in examples:
        if ex.content_id in seen_ids:
            dup_ids.append(ex.content_id)
            continue
        if ex.text in seen_text:
            dup_content.append(ex.content_id)
            continue
        seen_ids.add(ex.content_id)
        seen_text.add(ex.text)
        unique.append(ex)

    normalized_groups: dict[str, list[str]] = {}
    for ex in unique:
        normalized_groups.setdefault(normalize_text(ex.text), []).append(ex.content_id)
    near_dup_groups = {k: v for k, v in normalized_groups.items() if len(v) > 1}
    near_dup_examples = sum(len(v) for v in near_dup_groups.values())

    return DedupResult(
        unique=unique,
        duplicate_ids_removed=dup_ids,
        duplicate_content_removed=dup_content,
        normalized_duplicate_groups=len(near_dup_groups),
        normalized_duplicate_examples=near_dup_examples,
    )
