#!/usr/bin/env python3
"""make data-validate -- run data quality checks on the raw pool and write a
JSON report under benchmarks/results/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safelens.data.validation.report import build_validation_report
from safelens.utils.logging import configure_logging, get_logger

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "civil_comments" / "pool.jsonl"
METADATA_PATH = ROOT / "data" / "raw" / "civil_comments" / "metadata.json"
REPORT_PATH = ROOT / "benchmarks" / "results" / "data_validation_report.json"


def main() -> None:
    configure_logging()
    log = get_logger("safelens.scripts.validate_data")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    if not args.raw_path.exists():
        log.error("%s not found. Run `make data-download` first.", args.raw_path)
        raise SystemExit(1)

    metadata = json.loads(METADATA_PATH.read_text())
    dataset_version = (
        f"{metadata['dataset_name']}@{metadata['dataset_split']}:0-{metadata['actual_rows']}"
    )

    rows = [json.loads(line) for line in args.raw_path.read_text().splitlines() if line.strip()]
    report = build_validation_report(rows, dataset_version=dataset_version)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    log.info("validation report written to %s", args.report_path)
    log.info(
        "total=%d valid=%d malformed=%d invalid=%d dup_ids=%d dup_content=%d",
        report["total_rows"],
        report["valid_rows"],
        report["malformed_rows"],
        report["invalid_rows"],
        report["duplicate_ids_found"],
        report["duplicate_content_found"],
    )


if __name__ == "__main__":
    main()
