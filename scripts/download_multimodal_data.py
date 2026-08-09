#!/usr/bin/env python3
"""make multimodal-download -- ingest QCRI/Prop2Hate-Meme (CC-BY-NC-SA-4.0),
preserving the official train/dev/test split exactly as published."""

from __future__ import annotations

import argparse
from pathlib import Path

from safelens.data.multimodal.ingestion.prop2hate_meme import (
    DatasetAlreadyExistsError,
    download_prop2hate_meme,
)
from safelens.utils.logging import configure_logging, get_logger

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "multimodal" / "raw"


def main() -> None:
    configure_logging()
    log = get_logger("safelens.scripts.download_multimodal_data")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        result = download_prop2hate_meme(RAW_DIR, force=args.force)
    except DatasetAlreadyExistsError as exc:
        log.error(str(exc))
        raise SystemExit(1) from exc

    log.info("ingested split counts: %s", result.split_counts)
    log.info("processed dir: %s", result.processed_dir)
    log.info("metadata: %s", result.metadata_path)


if __name__ == "__main__":
    main()
