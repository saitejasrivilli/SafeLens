"""High-confidence false positive / false negative extraction for the
fusion model. Includes both text and image_path -- unlike the unimodal
Phase 5A/5B error analyses, the fusion model legitimately uses both
modalities, so both are legitimate to inspect when explaining its errors.
"""

from __future__ import annotations

from typing import Any

MAX_CHARS = 100


def _truncate(text: str) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS].rstrip() + "..."


def find_error_examples(
    example_ids: list[str],
    texts: list[str],
    image_paths: list[str],
    y_true: list[int],
    y_prob: list[float],
    threshold: float,
    top_k: int = 5,
) -> dict[str, Any]:
    rows = list(zip(example_ids, texts, image_paths, y_true, y_prob, strict=True))

    false_positives = [r for r in rows if r[3] == 0 and r[4] >= threshold]
    false_negatives = [r for r in rows if r[3] == 1 and r[4] < threshold]

    high_conf_fp = sorted(false_positives, key=lambda r: r[4], reverse=True)[:top_k]
    high_conf_fn = sorted(false_negatives, key=lambda r: r[4])[:top_k]

    def _row_to_dict(row: tuple[str, str, str, int, float]) -> dict[str, Any]:
        example_id, text, image_path, true_label, prob = row
        return {
            "example_id": example_id,
            "text_excerpt": _truncate(text),
            "image_path": image_path,
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
