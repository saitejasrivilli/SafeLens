"""Shared text normalization. Used identically by dedup, leakage, and (later)
training/serving preprocessing to keep training-serving skew out of scope."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = _PUNCT_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def content_id_for(text: str) -> str:
    """Deterministic content ID. civil_comments has no native record ID, so
    the ID is derived from the raw text — duplicate-ID detection is therefore
    equivalent to exact-duplicate-content detection for this dataset."""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
