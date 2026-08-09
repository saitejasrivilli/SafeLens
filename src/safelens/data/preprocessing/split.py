"""Deterministic, stratified, leakage-safe train/validation/test split.

civil_comments carries no per-record timestamp, so only a random
(seeded, stratified) split is implemented here. `SplitConfig` and the
per-example `timestamp` field on `ModerationExample` exist specifically so a
time-based split can be added later for datasets that do have timestamps,
without changing the data contract — do not fabricate timestamps to enable
that now.

Splitting happens at the granularity of normalized-text GROUPS, not
individual examples: every example whose normalized text matches another
example's is placed in the same split. This guarantees zero cross-split
leakage by construction, instead of relying on a leakage check to catch it
after the fact. Examples whose normalized text is empty (e.g. pure
punctuation/emoji, which collapse to "" and would otherwise be falsely
grouped together) are each treated as their own singleton group.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from safelens.data.preprocessing.normalize import normalize_text
from safelens.data.schema import ModerationExample

TOXIC_THRESHOLD = 0.5


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    train_frac: float = 0.7
    val_frac: float = 0.15
    # test_frac is implied: 1 - train_frac - val_frac


def _group_examples(examples: list[ModerationExample]) -> list[list[ModerationExample]]:
    groups: dict[str, list[ModerationExample]] = {}
    singleton_idx = 0
    for ex in examples:
        key = normalize_text(ex.text)
        if not key:
            key = f"__empty_normalized__{singleton_idx}"
            singleton_idx += 1
        groups.setdefault(key, []).append(ex)
    return list(groups.values())


def _group_stratify_key(group: list[ModerationExample]) -> int:
    return int(max(ex.labels.toxicity for ex in group) >= TOXIC_THRESHOLD)


def split_dataset(
    examples: list[ModerationExample], config: SplitConfig = SplitConfig()
) -> dict[str, list[ModerationExample]]:
    """Stratified, leakage-safe split, fixed seed. Same input + same config
    always yields the same three lists in the same order."""
    if not 0 < config.train_frac < 1 or not 0 < config.val_frac < 1:
        raise ValueError("train_frac and val_frac must be in (0, 1)")
    test_frac = 1 - config.train_frac - config.val_frac
    if test_frac <= 0:
        raise ValueError("train_frac + val_frac must be < 1")

    groups = _group_examples(examples)
    group_labels = [_group_stratify_key(g) for g in groups]

    train_groups, rest_groups, _, rest_labels = train_test_split(
        groups,
        group_labels,
        train_size=config.train_frac,
        random_state=config.seed,
        stratify=group_labels,
    )

    rest_val_share = config.val_frac / (config.val_frac + test_frac)
    val_groups, test_groups = train_test_split(
        rest_groups,
        train_size=rest_val_share,
        random_state=config.seed,
        stratify=rest_labels,
    )

    return {
        "train": [ex for g in train_groups for ex in g],
        "validation": [ex for g in val_groups for ex in g],
        "test": [ex for g in test_groups for ex in g],
    }
