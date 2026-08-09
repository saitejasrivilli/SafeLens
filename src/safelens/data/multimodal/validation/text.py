"""Text validation. Arabic text is used as-is -- never translated or
ASCII-normalized. Checks Unicode integrity, not content."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

REPLACEMENT_CHAR = "�"

# Arabic script + Arabic presentation forms, per Unicode block ranges.
_ARABIC_RANGES = [(0x0600, 0x06FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]


@dataclass(frozen=True)
class TextCheckResult:
    exists: bool
    non_empty: bool
    has_replacement_char: bool
    contains_arabic: bool
    error: str | None = None


def _contains_arabic(text: str) -> bool:
    return any(any(start <= ord(ch) <= end for start, end in _ARABIC_RANGES) for ch in text)


def check_text(text: object) -> TextCheckResult:
    if text is None:
        return TextCheckResult(
            exists=False,
            non_empty=False,
            has_replacement_char=False,
            contains_arabic=False,
            error="missing",
        )
    if not isinstance(text, str):
        return TextCheckResult(
            exists=True,
            non_empty=False,
            has_replacement_char=False,
            contains_arabic=False,
            error=f"not a string: {type(text).__name__}",
        )

    # NFC normalization check only -- never lowercase/strip-accents/ASCII-fold.
    normalized = unicodedata.normalize("NFC", text)
    stripped = normalized.strip()

    return TextCheckResult(
        exists=True,
        non_empty=bool(stripped),
        has_replacement_char=REPLACEMENT_CHAR in text,
        contains_arabic=_contains_arabic(text),
    )
