"""Complementarity analysis: compares per-example correctness (at threshold
0.5) across the three STANDALONE/ablation prediction sets to find direct
evidence of whether fusion captures information neither modality alone
does. Deterministic -- iterates the full test set in order, no sampling or
cherry-picking.
"""

from __future__ import annotations

from typing import Any


def _correct(y_true: int, y_prob: float, threshold: float = 0.5) -> bool:
    return int(y_prob >= threshold) == y_true


def analyze_complementarity(
    example_ids: list[str],
    y_true: list[int],
    text_only_probs: list[float],
    image_only_probs: list[float],
    multimodal_probs: list[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    text_correct = [_correct(t, p, threshold) for t, p in zip(y_true, text_only_probs, strict=True)]
    image_correct = [
        _correct(t, p, threshold) for t, p in zip(y_true, image_only_probs, strict=True)
    ]
    fusion_correct = [
        _correct(t, p, threshold) for t, p in zip(y_true, multimodal_probs, strict=True)
    ]

    both_wrong_fusion_right: list[str] = []
    text_right_image_wrong_fusion_wrong: list[str] = []
    image_right_text_wrong_fusion_wrong: list[str] = []
    all_correct: list[str] = []
    all_wrong: list[str] = []

    for i, eid in enumerate(example_ids):
        t, im, f = text_correct[i], image_correct[i], fusion_correct[i]
        if not t and not im and f:
            both_wrong_fusion_right.append(eid)
        elif t and not im and not f:
            text_right_image_wrong_fusion_wrong.append(eid)
        elif im and not t and not f:
            image_right_text_wrong_fusion_wrong.append(eid)
        elif t and im and f:
            all_correct.append(eid)
        elif not t and not im and not f:
            all_wrong.append(eid)

    n = len(example_ids)
    return {
        "threshold": threshold,
        "n": n,
        "counts": {
            "text_wrong_image_wrong_fusion_correct": len(both_wrong_fusion_right),
            "text_correct_image_wrong_fusion_wrong": len(text_right_image_wrong_fusion_wrong),
            "image_correct_text_wrong_fusion_wrong": len(image_right_text_wrong_fusion_wrong),
            "all_three_correct": len(all_correct),
            "all_three_wrong": len(all_wrong),
        },
        "example_ids": {
            "text_wrong_image_wrong_fusion_correct": both_wrong_fusion_right,
            "text_correct_image_wrong_fusion_wrong": text_right_image_wrong_fusion_wrong,
            "image_correct_text_wrong_fusion_wrong": image_right_text_wrong_fusion_wrong,
        },
        "accuracy_at_threshold": {
            "text_only": sum(text_correct) / n if n else 0.0,
            "image_only": sum(image_correct) / n if n else 0.0,
            "multimodal": sum(fusion_correct) / n if n else 0.0,
        },
    }
