"""Builds the Prop2Hate-Meme data quality validation report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from safelens.data.multimodal.schema import HATE_LABEL_NAMES, MultimodalExample
from safelens.data.multimodal.validation.images import check_image
from safelens.data.multimodal.validation.text import check_text


def _label_distribution(examples: list[MultimodalExample]) -> dict[str, Any]:
    counts = {name: 0 for name in HATE_LABEL_NAMES}
    for ex in examples:
        counts[ex.hate_label_name] += 1
    total = len(examples)
    positive = counts["hateful"]
    return {
        "counts": counts,
        "total": total,
        "positive_rate": positive / total if total else 0.0,
    }


def load_valid_examples(processed_dir: Path) -> dict[str, list[MultimodalExample]]:
    """Parses each split's JSONL into schema-validated examples, silently
    skipping malformed rows (the validation report is the place those get
    surfaced) -- used by dedup/leakage, which need real objects, not just
    counts."""
    result: dict[str, list[MultimodalExample]] = {}
    for split_name in ("train", "dev", "test"):
        jsonl_path = processed_dir / f"{split_name}.jsonl"
        if not jsonl_path.exists():
            result[split_name] = []
            continue
        examples = []
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                examples.append(MultimodalExample(**json.loads(line)))
            except ValidationError:
                continue
        result[split_name] = examples
    return result


def build_validation_report(processed_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"splits": {}}
    total_images_checked = 0
    total_valid_images = 0
    total_corrupted_images = 0
    total_missing_images = 0
    total_arabic_rows = 0
    total_rows = 0

    for split_name in ("train", "dev", "test"):
        jsonl_path = processed_dir / f"{split_name}.jsonl"
        if not jsonl_path.exists():
            report["splits"][split_name] = {"error": f"{jsonl_path} not found"}
            continue

        rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]

        valid: list[MultimodalExample] = []
        malformed: list[dict[str, Any]] = []
        image_issues: list[dict[str, Any]] = []
        text_issues: list[dict[str, Any]] = []
        arabic_count = 0

        for row in rows:
            text_result = check_text(row.get("text"))
            if not text_result.non_empty or text_result.has_replacement_char:
                text_issues.append(
                    {
                        "example_id": row.get("example_id"),
                        "reason": text_result.error or "invalid text",
                    }
                )
            if text_result.contains_arabic:
                arabic_count += 1

            try:
                example = MultimodalExample(**row)
            except ValidationError as exc:
                malformed.append({"example_id": row.get("example_id"), "reasons": [str(exc)]})
                continue

            image_result = check_image(processed_dir / example.image_path)
            total_images_checked += 1
            if not image_result.exists:
                total_missing_images += 1
                image_issues.append({"example_id": example.example_id, "reason": "missing"})
            elif not image_result.decodable:
                total_corrupted_images += 1
                image_issues.append(
                    {"example_id": example.example_id, "reason": image_result.error}
                )
            else:
                total_valid_images += 1

            valid.append(example)

        total_arabic_rows += arabic_count
        total_rows += len(rows)

        report["splits"][split_name] = {
            "total_rows": len(rows),
            "valid_rows": len(valid),
            "malformed_rows": len(malformed),
            "malformed_examples": malformed[:20],
            "text_issues": len(text_issues),
            "text_issue_examples": text_issues[:20],
            "arabic_text_fraction": arabic_count / len(rows) if rows else 0.0,
            "image_issues": len(image_issues),
            "image_issue_examples": image_issues[:20],
            "label_distribution": {
                "hate_label": _label_distribution(valid),
            },
        }

    report["image_validation_summary"] = {
        "total_checked": total_images_checked,
        "valid": total_valid_images,
        "corrupted": total_corrupted_images,
        "missing": total_missing_images,
    }
    report["arabic_text_fraction_overall"] = total_arabic_rows / total_rows if total_rows else 0.0

    return report
