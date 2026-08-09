#!/usr/bin/env python3
"""make data-download -- fetch a deterministic prototype slice of
google/civil_comments into data/raw/civil_comments/."""

from __future__ import annotations

import argparse
from pathlib import Path

from safelens.data.ingestion.civil_comments import (
    DatasetAlreadyExistsError,
    download_civil_comments,
)
from safelens.utils.logging import configure_logging, get_logger

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "civil_comments"


def main() -> None:
    configure_logging()
    log = get_logger("safelens.scripts.download_data")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=8000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        result = download_civil_comments(RAW_DIR, n_rows=args.n_rows, force=args.force)
    except DatasetAlreadyExistsError as exc:
        log.error(str(exc))
        raise SystemExit(1) from exc

    log.info(
        "downloaded %d rows -> %s (metadata: %s)",
        result.num_rows,
        result.raw_path,
        result.metadata_path,
    )


if __name__ == "__main__":
    main()
