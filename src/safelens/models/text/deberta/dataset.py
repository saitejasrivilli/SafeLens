"""Deterministic tokenization + torch Dataset wrapper.

Reuses safelens.models.text.data.load_split (Phase 2 processed data, read
only) and safelens.models.text.data.binarize (same ground-truth threshold
semantics as the Phase 3 baseline) so both experiments start from identical
inputs.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class ToxicityDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        content_ids: list[str],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
    ):
        # Tokenization is deterministic: fixed truncation/padding/max_length,
        # no randomness, same output given the same tokenizer + inputs.
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.content_ids = content_ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
