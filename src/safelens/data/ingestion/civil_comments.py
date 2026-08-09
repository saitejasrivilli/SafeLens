"""Deterministic ingestion of a prototype subset of google/civil_comments.

Source: https://huggingface.co/datasets/google/civil_comments
License: CC0-1.0 (public domain dedication)

The full dataset has 1,804,874 train rows (~595MB). For a local M2 prototype
we ingest a fixed-offset, fixed-length slice of the "train" split via the
Hugging Face datasets-server rows API. Row order for a given dataset revision
is stable (backed by fixed parquet files), so offset=0, length=N always
returns the same N rows -> reruns are reproducible without re-hashing the
full 595MB corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

DATASET_NAME = "google/civil_comments"
DATASET_CONFIG = "default"
DATASET_LICENSE = "cc0-1.0"
SOURCE_URL = "https://huggingface.co/datasets/google/civil_comments"
ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100


class DatasetAlreadyExistsError(RuntimeError):
    pass


@dataclass(frozen=True)
class IngestionResult:
    raw_path: Path
    metadata_path: Path
    num_rows: int


def _fetch_page(split: str, offset: int, length: int) -> list[dict[str, Any]]:
    params: dict[str, str | int] = {
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "split": split,
        "offset": offset,
        "length": length,
    }
    resp = requests.get(ROWS_API, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(f"datasets-server error: {payload['error']}")
    return [r["row"] for r in payload["rows"]]


def fetch_rows(split: str, n_rows: int, page_size: int = PAGE_SIZE) -> list[dict[str, Any]]:
    """Fetch the first n_rows of `split`, in fixed deterministic offset order."""
    rows: list[dict[str, Any]] = []
    offset = 0
    while len(rows) < n_rows:
        length = min(page_size, n_rows - len(rows))
        batch = _fetch_page(split, offset, length)
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    return rows[:n_rows]


def download_civil_comments(
    output_dir: Path,
    n_rows: int = 8000,
    split: str = "train",
    force: bool = False,
) -> IngestionResult:
    """Download a deterministic row slice and write raw JSONL + metadata.

    Never overwrites existing raw data unless force=True.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "pool.jsonl"
    metadata_path = output_dir / "metadata.json"

    if raw_path.exists() and not force:
        raise DatasetAlreadyExistsError(
            f"{raw_path} already exists. Use force=True to re-download."
        )

    rows = fetch_rows(split, n_rows)

    with raw_path.open("w") as f:
        for idx, row in enumerate(rows):
            f.write(json.dumps({"row_idx": idx, **row}) + "\n")

    metadata = {
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": split,
        "license": DATASET_LICENSE,
        "source_url": SOURCE_URL,
        "retrieval_date": datetime.now(UTC).isoformat(),
        "requested_rows": n_rows,
        "actual_rows": len(rows),
        "offset": 0,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return IngestionResult(raw_path=raw_path, metadata_path=metadata_path, num_rows=len(rows))
