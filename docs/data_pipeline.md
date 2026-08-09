# Data Pipeline

**Note:** this document covers the Phase 2-4 `civil_comments` (English text)
pipeline only. Phase 5 uses a separate dataset (`QCRI/Prop2Hate-Meme`,
Arabic image+text memes) with its own pipeline, manifest, and documentation
— see `docs/multimodal_design.md`. The two are not combined.

Phase 5's leakage-handling approach differs from Phase 2's: where Phase 2's
`civil_comments` split had zero leakage by construction (grouped, stratified
split — see §Leakage prevention above), Phase 5's *official* raw split
actually contains 2 minor cross-split caption duplicates. Rather than
editing the official raw split, Phase 5 preserves it immutably and produces
a separate leakage-clean *processed* training split instead — see
`docs/multimodal_design.md` §10 for the full remediation rule and results.

## Data source

**Dataset:** [`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments) on Hugging Face.

- **Origin:** public comments from the Civil Comments platform (a commenting plugin used
  on ~50 English-language news sites, 2015–2017). Released as an open archive for research
  when the platform shut down.
- **License:** `cc0-1.0` (public domain dedication) — verified via the Hugging Face dataset
  API (`cardData.license`), no usage restriction.
- **Full corpus size:** 1,804,874 train rows / 97,320 validation rows / 97,320 test rows
  (per HF `dataset_info`), ~595MB train split.
- **Supported labels:** 7 continuous per-comment scores in `[0, 1]`, each the fraction of
  human annotators who flagged that attribute: `toxicity`, `severe_toxicity`, `obscene`,
  `threat`, `insult`, `identity_attack`, `sexual_explicit`. These are the *only* labels the
  dataset provides — SafeLens does not invent additional categories on top of them (see
  `docs/policy_taxonomy.md` for how they map to, and fall short of, the target taxonomy).
- **Why this dataset:** real human-annotated moderation-relevant text at meaningful scale,
  permissive license, multi-attribute (supports multi-label classification later), and
  directly accessible via the Hugging Face `datasets-server` REST API without gated access
  or an API key — reproducible from a fresh machine with just a URL.

## Prototype scope (honesty note)

The full corpus (1.8M rows, 595MB) is not ingested. For a local M2 prototype, SafeLens
ingests a **deterministic 8,000-row slice** (`offset=0, length=8000`) of the `train` split.
Row order in the underlying parquet files is fixed for a given dataset revision, so this
slice is exactly reproducible across machines and reruns — it is a fixed subset, not a
random sample. All counts reported below are for this 8,000-row slice, not the full corpus.

## Data contract

Defined in `src/safelens/data/schema.py` using Pydantic:

```
ModerationExample:
  content_id: str        # sha256(text)[:16] — civil_comments has no native ID
  text: str               # non-empty after stripping whitespace
  image_ref: str | None    # unused for this text-only dataset
  labels: LabelScores      # 7 scores, each constrained to [0.0, 1.0]
  source: str              # "google/civil_comments"
  timestamp: str | None    # None — dataset has no per-record timestamp (not fabricated)
  dataset_version: str     # e.g. "google/civil_comments@train:0-8000"
```

Invalid records (missing fields, non-numeric labels, out-of-range scores, empty text) are
rejected, not silently coerced.

## Preprocessing

1. **Ingestion** (`scripts/download_data.py` → `src/safelens/data/ingestion/civil_comments.py`):
   fetches the deterministic row slice via the HF `datasets-server` rows API, writes
   `data/raw/civil_comments/pool.jsonl` + `metadata.json`. Refuses to overwrite existing
   raw data unless `--force` is passed.
2. **Schema validation + mapping** (`src/safelens/data/mapping.py`): each raw row is mapped
   to `ModerationExample`, distinguishing *malformed* (missing/non-numeric fields) from
   *invalid* (schema-valid types but out-of-range values, e.g. score > 1.0).
3. **Deduplication** (`src/safelens/data/preprocessing/dedup.py`): exact duplicate IDs and
   exact duplicate text are removed (keeping first occurrence, deterministic). Because
   `content_id` is a hash of `text`, ID-duplicates and content-duplicates coincide for this
   dataset. Normalized-text near-duplicates (case/punctuation/whitespace differences) are
   detected and counted, but not removed — near-identical comments are still distinct
   moderation examples.
4. **Leakage-safe splitting** (`src/safelens/data/preprocessing/split.py`): rather than
   splitting first and checking for leakage after, examples are grouped by normalized text
   *before* splitting, and each group is assigned to exactly one split — the split is
   leakage-free by construction, not by post-hoc detection. Examples whose normalized text
   is empty (pure punctuation/emoji) are treated as singleton groups, since grouping all of
   them together would falsely conflate unrelated content. Splitting is stratified on
   binarized `toxicity >= 0.5` per group, `random_state=42`, via `sklearn.train_test_split`.
5. **Leakage check** (`src/safelens/data/preprocessing/leakage.py`) still runs after
   splitting, as an independent verification (not the sole leakage-prevention mechanism):
   checks for duplicate `content_id` and duplicate normalized text across every pair of
   splits.
6. **Manifest** (`src/safelens/data/manifest.py`): records dataset identity, license,
   retrieval date, raw/processed data hashes (sha256), record counts, label distribution,
   split sizes, preprocessing version, and random seed.

## Time-based split (not implemented — documented limitation)

`google/civil_comments` has no per-comment timestamp field (only the aggregate description
that comments span 2015–2017). SafeLens does not fabricate timestamps to force a time-based
split. The `timestamp: str | None` field on `ModerationExample` and the general split
architecture are designed so a time-based split can be added for a future dataset that does
carry per-record timestamps, without changing the data contract.

## Reproducibility

Running `download` (fixed offset/length) then `prepare` (fixed seed=42) twice on the same
raw data produces identical processed-data hashes, split sizes, and manifest values except
for `retrieval_date` (a real wall-clock timestamp of when the download ran). Verified by
running the full pipeline twice end-to-end — see the "Actual results" report below.

## Limitations

- 8,000-row prototype slice, not the full 1.8M-row corpus — do not compare metrics computed
  on this slice against benchmarks using the full dataset.
- Labels are soft annotator-agreement fractions, not ground-truth binary labels — any
  downstream classifier training must pick and document a threshold explicitly (Phase 3+).
- No timestamps → no time-based (distribution-shift) evaluation split is possible with this
  dataset.
- No SPAM or SELF_HARM signal in this dataset — see `docs/policy_taxonomy.md`.
- No image data — this dataset only supports the text pipeline, not the vision/multimodal
  phases.

## Actual results (this run)

See `benchmarks/results/data_validation_report.json` for the raw validation report and
`data/manifests/civil_comments_manifest.json` for the generated manifest. Summary as
actually measured on 2026-08-09:

- Rows fetched: 8,000 / 8,000 requested (100%)
- Schema-valid: 8,000 / 8,000
- Malformed: 0, Invalid (out-of-range): 0
- Exact duplicate IDs/content removed: 18
- Normalized near-duplicates detected (not removed): 37 examples across several groups
- Unique records after dedup: 7,982
- Split sizes: train 5,588 / validation 1,197 / test 1,197
- Leakage check: clean (0 ID overlaps, 0 normalized-text overlaps) across all split pairs
