"""High-confidence false positive / false negative extraction for the
image-only model. Deliberately excludes text -- the image-only experiment's
primary error analysis must not use the text field to explain predictions.
"""

from __future__ import annotations

from typing import Any


def find_error_examples(
    example_ids: list[str],
    image_paths: list[str],
    y_true: list[int],
    y_prob: list[float],
    threshold: float,
    top_k: int = 5,
) -> dict[str, Any]:
    rows = list(zip(example_ids, image_paths, y_true, y_prob, strict=True))

    false_positives = [r for r in rows if r[2] == 0 and r[3] >= threshold]
    false_negatives = [r for r in rows if r[2] == 1 and r[3] < threshold]

    high_conf_fp = sorted(false_positives, key=lambda r: r[3], reverse=True)[:top_k]
    high_conf_fn = sorted(false_negatives, key=lambda r: r[3])[:top_k]

    def _row_to_dict(row: tuple[str, str, int, float]) -> dict[str, Any]:
        example_id, image_path, true_label, prob = row
        return {
            "example_id": example_id,
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
