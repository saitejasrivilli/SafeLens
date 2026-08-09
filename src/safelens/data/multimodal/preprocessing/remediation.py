"""Leakage remediation for the processed training split ONLY.

The official raw train/dev/test split (`data/multimodal/raw/`) is never
modified -- this module produces a separate, leakage-clean *processed*
training set by dropping training examples whose exact caption text also
appears in dev or test. Dev and test are never touched.

Rule (exact text match only, no normalization, no semantic dedup -- the
same exact-text definition already used by
`safelens.data.multimodal.preprocessing.leakage.check_leakage`):
  if train_example.text in (dev texts) union (test texts):
      remove train_example from the processed training set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from safelens.data.multimodal.schema import MultimodalExample


@dataclass(frozen=True)
class RemediationResult:
    clean_train: list[MultimodalExample]
    removed_examples: list[MultimodalExample] = field(default_factory=list)

    @property
    def original_count(self) -> int:
        return len(self.clean_train) + len(self.removed_examples)

    @property
    def removed_count(self) -> int:
        return len(self.removed_examples)

    @property
    def final_count(self) -> int:
        return len(self.clean_train)


def remediate_train_leakage(
    train: list[MultimodalExample],
    dev: list[MultimodalExample],
    test: list[MultimodalExample],
) -> RemediationResult:
    banned_texts = {ex.text for ex in dev} | {ex.text for ex in test}

    clean_train: list[MultimodalExample] = []
    removed: list[MultimodalExample] = []
    for ex in train:  # order-preserving, deterministic
        if ex.text in banned_texts:
            removed.append(ex)
        else:
            clean_train.append(ex)

    return RemediationResult(clean_train=clean_train, removed_examples=removed)
