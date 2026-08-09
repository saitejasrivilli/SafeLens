"""Builds a data quality validation report from raw civil_comments rows."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from safelens.data.mapping import MalformedRowError, civil_comments_row_to_example
from safelens.data.preprocessing.dedup import deduplicate
from safelens.data.schema import ModerationExample

LABEL_NAMES = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]


def build_validation_report(
    raw_rows: list[dict[str, Any]], dataset_version: str
) -> dict[str, Any]:
    valid: list[ModerationExample] = []
    malformed: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []

    for row in raw_rows:
        try:
            example = civil_comments_row_to_example(row, dataset_version=dataset_version)
        except MalformedRowError as exc:
            malformed.append({"row_idx": exc.row_idx, "reasons": exc.reasons})
            continue
        except ValidationError as exc:
            invalid.append({"row_idx": row.get("row_idx"), "reasons": [str(exc)]})
            continue
        valid.append(example)

    dedup_result = deduplicate(valid)

    label_distribution = {}
    for name in LABEL_NAMES:
        scores = [getattr(ex.labels, name) for ex in valid]
        label_distribution[name] = {
            "mean": sum(scores) / len(scores) if scores else 0.0,
            "fraction_ge_0.5": (
                sum(1 for s in scores if s >= 0.5) / len(scores) if scores else 0.0
            ),
        }

    return {
        "total_rows": len(raw_rows),
        "valid_rows": len(valid),
        "malformed_rows": len(malformed),
        "invalid_rows": len(invalid),
        "malformed_examples": malformed[:20],
        "invalid_examples": invalid[:20],
        "duplicate_ids_found": len(dedup_result.duplicate_ids_removed),
        "duplicate_content_found": len(dedup_result.duplicate_content_removed),
        "normalized_duplicate_groups": dedup_result.normalized_duplicate_groups,
        "normalized_duplicate_examples": dedup_result.normalized_duplicate_examples,
        "unique_valid_rows_after_dedup": len(dedup_result.unique),
        "label_distribution": label_distribution,
    }
