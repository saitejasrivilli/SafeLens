# data/

This directory holds dataset artifacts. **Raw and processed data are not committed to
Git** (see `.gitignore`) — only manifests, metadata, and this README are tracked.

```
data/
├── raw/civil_comments/
│   ├── pool.jsonl        # gitignored — deterministic 8,000-row raw slice
│   └── metadata.json     # gitignored — retrieval date, source, license, row count
├── processed/civil_comments/
│   ├── train.jsonl       # gitignored — validated ModerationExample records
│   ├── validation.jsonl  # gitignored
│   └── test.jsonl        # gitignored
└── manifests/
    └── civil_comments_manifest.json   # tracked — machine-readable dataset manifest
```

## Reproducing the data locally

```
make data-download   # fetch deterministic raw slice (never overwrites existing raw data)
make data-validate    # run data quality checks -> benchmarks/results/data_validation_report.json
make data-prepare     # dedup + leakage-safe split + manifest -> data/processed/, data/manifests/
make data-test        # run the data-pipeline unit tests
```

See `docs/data_pipeline.md` for the full pipeline design, dataset source/license, and
measured results.
