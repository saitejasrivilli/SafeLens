"""Maps a raw civil_comments row (dict) into the SafeLens data contract."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from safelens.data.preprocessing.normalize import content_id_for
from safelens.data.schema import LabelScores, ModerationExample

REQUIRED_RAW_FIELDS = [
    "text",
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]


class MalformedRowError(ValueError):
    def __init__(self, row_idx: Any, reasons: list[str]):
        self.row_idx = row_idx
        self.reasons = reasons
        super().__init__(f"row {row_idx} malformed: {reasons}")


def civil_comments_row_to_example(
    row: dict[str, Any], dataset_version: str, source: str = "google/civil_comments"
) -> ModerationExample:
    """Raises MalformedRowError (missing/wrong-typed fields) or pydantic
    ValidationError (out-of-range values, empty text) on invalid input."""
    missing = [f for f in REQUIRED_RAW_FIELDS if f not in row or row[f] is None]
    if missing:
        raise MalformedRowError(row.get("row_idx"), [f"missing field: {f}" for f in missing])

    text = row["text"]
    if not isinstance(text, str):
        raise MalformedRowError(row.get("row_idx"), ["text is not a string"])

    try:
        label_values = {name: float(row[name]) for name in REQUIRED_RAW_FIELDS if name != "text"}
    except (TypeError, ValueError) as exc:
        raise MalformedRowError(row.get("row_idx"), [f"non-numeric label: {exc}"]) from exc

    # LabelScores/ModerationExample ValidationError (e.g. out-of-range score,
    # empty text) is a distinct failure mode from malformed input and
    # propagates to the caller uncaught.
    labels = LabelScores(**label_values)
    return ModerationExample(
        content_id=content_id_for(text),
        text=text,
        image_ref=None,
        labels=labels,
        source=source,
        timestamp=None,
        dataset_version=dataset_version,
    )


__all__ = ["civil_comments_row_to_example", "MalformedRowError", "ValidationError"]
