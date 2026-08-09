"""Exact-duplicate image detection via content hash. No perceptual/semantic
hashing -- exact duplicates only, per the same conservative approach used
in the Phase 2 text pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path


def image_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
