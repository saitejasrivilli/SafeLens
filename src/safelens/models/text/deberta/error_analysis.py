"""Extracts high-confidence false positive / false negative examples for
qualitative error analysis. Text is truncated for brevity -- civil_comments
is CC0-licensed public data, but full raw comments are not appropriate to
reproduce at length in documentation."""

from __future__ import annotations

from typing import Any

MAX_CHARS = 100


def _truncate(text: str) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS].rstrip() + "..."


def find_error_examples(
    content_ids: list[str],
    texts: list[str],
    y_true: list[int],
    y_prob: list[float],
    threshold: float,
    top_k: int = 5,
) -> dict[str, Any]:
    rows = list(zip(content_ids, texts, y_true, y_prob, strict=True))

    false_positives = [r for r in rows if r[2] == 0 and r[3] >= threshold]
    false_negatives = [r for r in rows if r[2] == 1 and r[3] < threshold]

    # "High-confidence" = furthest from the threshold on the wrong side.
    high_conf_fp = sorted(false_positives, key=lambda r: r[3], reverse=True)[:top_k]
    high_conf_fn = sorted(false_negatives, key=lambda r: r[3])[:top_k]

    def _row_to_dict(row: tuple[str, str, int, float]) -> dict[str, Any]:
        cid, text, true_label, prob = row
        return {
            "content_id": cid,
            "text_excerpt": _truncate(text),
            "true_label": true_label,
            "predicted_probability": round(prob, 4),
        }

    return {
        "threshold": threshold,
        "total_false_positives": len(false_positives),
        "total_false_negatives": len(false_negatives),
        "high_confidence_false_positives": [_row_to_dict(r) for r in high_conf_fp],
        "high_confidence_false_negatives": [_row_to_dict(r) for r in high_conf_fn],
    }
