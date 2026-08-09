# Multimodal Design — Phase 5 (Data Pipeline)

**Phase 5 uses Prop2Hate-Meme as a separate multimodal experiment. It is not
combined with the Civil Comments experiments (Phase 2-4).** Different
dataset, different language, different modality, its own split, its own
manifest, its own model track under `src/safelens/data/multimodal/` and
(later) `src/safelens/models/{vision,multimodal}/`.

## Status

Data pipeline only. No CLIP, no text encoder, no fusion model implemented
yet — this document covers dataset access, license, schema, validation, and
leakage findings for the ingestion stage.

## 1. Why Prop2Hate-Meme

Selected after two other candidates (Hateful Memes, HarMeme/MultiOFF) were
ruled out — Hateful Memes' DrivenData competition is closed with no
legitimate re-access path; HarMeme and MultiOFF's actual dataset *content*
(not just their papers/code) had no authors'-stated, authoritative license.
Prop2Hate-Meme is the first candidate found where the dataset content itself
(images + text + annotations, not just the paper) carries an explicit
license declared by the original creators in their own repository. See prior
research turns for the full comparison; this document only covers the
dataset actually selected.

## 2. Dataset source

`https://huggingface.co/datasets/QCRI/Prop2Hate-Meme` — Qatar Computing
Research Institute (QCRI). Extends the `QCRI/ArMeme` corpus (EMNLP 2024,
arXiv:2406.03916) with fine- and coarse-grained hate-speech annotations
(arXiv:2409.07246).

## 3. License

**CC-BY-NC-SA-4.0**, declared directly in the dataset repository's own YAML
front-matter (`license: cc-by-nc-sa-4.0`) — verified from the authoritative
source itself (`huggingface.co/datasets/QCRI/Prop2Hate-Meme/raw/main/README.md`
and the HF Hub API `cardData.license` field), not inferred from a paper or a
secondary description.

**What this means for SafeLens:**
- Non-commercial use only. This project is a non-commercial research/portfolio
  project — SafeLens is not presented as, and must not be presented as, a
  commercial product built on this data.
- ShareAlike: any redistributed *derivative dataset* must carry the same
  license. This does not restrict the license of SafeLens's own code.
- Attribution required (§4).
- The raw dataset (images, parquet files, extracted JPEGs) is **never
  committed to Git** — see §10 of the Phase 5 data-pipeline instructions and
  `.gitignore` (`data/multimodal/raw/`, `data/multimodal/processed/`).

## 4. Attribution

> Prop2Hate-Meme dataset by QCRI (Qatar Computing Research Institute),
> extending the ArMeme corpus with hate-speech annotations.
> https://arxiv.org/abs/2409.07246 (Prop2Hate-Meme) /
> https://arxiv.org/abs/2406.03916 (ArMeme, base corpus).

## 5. Dataset size (measured, not assumed)

**3,061 total examples** — train 2,143 / dev 312 / test 606. Matches the
officially documented split exactly (verified against the dataset's own
`dataset_info` split metadata during ingestion — see
`data/multimodal/manifests/prop2hate_meme_manifest.json`).

## 6. Official split

Preserved exactly as published — `scripts/download_multimodal_data.py` loads
the dataset via the `datasets` library's own `train`/`dev`/`test` split
objects and writes one JSONL per split with no reshuffling, resplitting, or
merging.

## 7. Labels

- `hate_label`: binary, `not-hateful` (0) / `hateful` (1). **Primary Phase 5
  target.**
- `hate_fine_grained_label`: 10-way (`sarcasm`, `humor`, `inciting_violence`,
  `mocking`, `other`, `exclusion`, `dehumanizing`, `contempt`, `inferiority`,
  `slurs`) — preserved in processed metadata for later analysis, **not
  trained on in Phase 5's first pass**.
- `prop_label`: binary propaganda/not-propaganda (inherited from the base
  ArMeme corpus) — preserved, **not trained on in Phase 5's first pass**.

**Measured label distribution** (not assumed):

| split | hateful | not-hateful | total | positive rate |
|---|---|---|---|---|
| train | 213 | 1,930 | 2,143 | 9.94% |
| dev | 31 | 281 | 312 | 9.94% |
| test | 154 | 452 | 606 | **25.41%** |

**Important, measured finding:** the test split's positive rate (25.4%) is
**substantially higher** than train/dev (~9.9% each) — a real, measured
distribution shift in the official split, not an assumption. This has direct
consequences for later threshold selection (a threshold tuned on
dev's ~10% positive rate should not be assumed to transfer cleanly to test's
~25%) and must be accounted for explicitly when Phase 5A-5C are implemented,
not discovered as a surprise later.

## 8. Arabic-language characteristic

100% of text fields contain Arabic-script characters (measured:
`arabic_text_fraction_overall = 1.0` in the validation report). Text is
preserved exactly as provided — no translation, no ASCII normalization, no
stripping of Arabic characters (`src/safelens/data/multimodal/validation/text.py`
explicitly checks for and reports Arabic-script presence and Unicode
replacement-character corruption, and only applies NFC normalization, never
transliteration or lowercasing). This is why Phase 4's English-only
`microsoft/deberta-v3-small` cannot be reused as-is for Phase 5's text
branch — see `docs/model_design.md` for the Arabic/multilingual encoder
options under evaluation.

## 9. Validation (measured results)

- **Schema:** 3,061/3,061 rows schema-valid (0 malformed) across all three
  splits.
- **Images:** 3,061/3,061 valid, decodable, positive-dimension JPEGs. 0
  corrupted, 0 missing.
- **Text:** 0 empty-text rows, 0 Unicode replacement-character corruption,
  100% contain Arabic script.

## 10. Leakage checks and remediation

Checked: duplicate IDs, duplicate text, duplicate image hashes (exact
content hash, not perceptual), duplicate text+image pairs — within each
split and across every split pair (train↔dev, train↔test, dev↔test).

**A. Official raw split is preserved, unmodified, and is NOT leakage-free.**
`data/multimodal/raw/prop2hate_meme/{train,dev,test}.jsonl` is exactly what
was downloaded — 2 cross-split exact-caption text overlaps exist in it (one
`train↔dev`, one `train↔test`; each caption appears on a **different image**
in each split — 0 image-hash overlaps, 0 full text+image pair overlaps, 0 ID
overlaps). This raw split is never edited, per instructions — it remains
the immutable source of truth, leakage warts and all.

**B. A separate processed training split removes the exact-caption overlap.**
`scripts/remediate_multimodal_leakage.py` produces
`data/multimodal/processed/prop2hate_meme/{train,dev,test}.jsonl`:
- **Rule:** a training example is dropped if its caption text exactly
  matches any dev or test caption text (exact string match only — no
  normalization beyond what the validation report already applies, no
  semantic/fuzzy deduplication).
- **Removed:** 2 of 2,143 original training examples.
- **Final processed training count:** 2,141.
- Both removed examples were confirmed via manual inspection to be short,
  proverb/phrase-style captions ("A ship won't sink if you are its
  captain"-type sayings) — the kind of caption plausibly reused across
  unrelated memes, not evidence of the dataset containing true full-record
  duplicates.

**C. Dev and test are copied into the processed directory unchanged** — same
312 / 606 rows, same image references, same labels. Nothing about dev/test
is modified by remediation; they are duplicated into the processed
directory purely so training code has one consistent place to read all
three splits from.

**Post-remediation verification:** re-running the leakage check on
`{clean_train, dev, test}` confirms `is_clean=True` — zero ID, text, image-hash,
or pair overlaps remain between the processed training split and dev/test.
Full detail: `benchmarks/results/multimodal/leakage_remediation_report.json`.

**Why the raw split stays untouched:** the raw split is the authoritative,
verifiable record of exactly what QCRI published — auditability and
reproducibility require it never be silently edited. Leakage remediation is
a *modeling-input* decision (don't let the text encoder see an evaluation
caption during training), which belongs in a separate, clearly-labeled
processed artifact, not in the raw record.

**The dataset is not "perfectly leakage-free" as a blanket claim** — the raw
official split still contains the 2 overlaps documented above. Only the
processed training split (train vs. dev, train vs. test) has been verified
clean.

## 10b. Test-set distribution shift (measured, no cause asserted)

| split | positive rate |
|---|---|
| train (raw) | 9.94% |
| train (leakage-clean, processed) | 9.95% |
| dev | 9.94% |
| test | **25.41%** |

Absolute difference (test − train): **+15.47 percentage points**. Relative
ratio (test / train): **2.56×** — test's hate-label prevalence is more than
double train/dev's.

**No authoritative explanation exists.** The official dataset README
(fetched directly from `huggingface.co/datasets/QCRI/Prop2Hate-Meme`) labels
the HF `test` config split as **"Dev-Test Split (`dev_test`)"** in its own
per-split statistics tables, but states no methodology for how that split
was constructed relative to train/dev, and gives no reason for the
prevalence difference. This is recorded as an **observed distribution
shift**, not an inferred cause — no explanation has been invented.

**Consequence for evaluation (binding for Phase 5A-5C):** a decision
threshold selected on dev (only 31 positive examples, ~9.9% prevalence)
should not be assumed to transfer to test (~25.4% prevalence). See
`docs/evaluation.md` for the resulting Phase 5 evaluation policy
(PR-AUC as the primary threshold-independent metric; fixed-threshold and
validation-selected-threshold results reported separately and labeled;
test never used to select a threshold).

## 11. Limitations

- 3,061 examples total — above the ">2,000" minimum but below the ">5,000"
  strong preference threshold discussed during dataset selection.
- Test-set positive rate (25.4%) differs substantially from train/dev
  (~9.9%) — a real distribution shift to account for, not an assumption.
- Minor cross-split text leakage in the **raw** official split (2 short
  captions reused on different images) — resolved for the **processed**
  training split via exact-caption remediation (§10); the raw split itself
  is intentionally left as-is, per instructions.
- Test-set hate-label prevalence (25.4%) is 2.56× train/dev's (~9.9%), with
  no authoritative explanation found in the dataset's own documentation
  (§10b) — binds all Phase 5 threshold-selection decisions to a
  validation-only, never-test discipline (`docs/evaluation.md`).
- Arabic-only text — requires a non-English text encoder for Phase 5B/5C,
  meaning Phase 4's DeBERTa-v3-small infrastructure cannot be reused
  directly for the text branch (only the general module *pattern*, not the
  specific model).
- `hate_fine_grained_label` and `prop_label` are preserved but unused in this
  phase — any future multi-task or multi-label extension needs its own
  explicit design pass, not a silent repurposing of this pipeline's output.
- CC-BY-NC-SA-4.0 restricts this work to non-commercial use — explicitly
  documented here so it isn't discovered as a surprise later if scope ever
  expands toward a commercial framing.

## 12. Phase 5A — image-only baseline (complete)

Frozen `openai/clip-vit-base-patch32` image encoder (revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`) + a small trained classification
head (Linear→ReLU→Dropout→Linear). **No text was used anywhere in this
model** — enforced in code (`extract_embeddings` never reads `.text`) and
verified by a test using a duck-typed stand-in whose `.text` attribute
raises if ever accessed.

Full results, error analysis, and limitations are in `docs/evaluation.md`
and `docs/experiments.md`. This is a baseline result only — **it does not
establish or claim any multimodal benefit**; that comparison is only
possible once Phase 5B (text-only) and Phase 5C (fusion) exist on the same
split.

## 13. Phase 5A visual error analysis (pixel-level, separate from metadata)

Before Phase 5B, a small deterministic manual review of the same 5
high-confidence false positives and 5 high-confidence false negatives
(threshold=0.6) was performed by actually viewing the image pixels — not
inferring from filenames, and not using the caption text. Full findings:
`benchmarks/results/multimodal/image_only/visual_error_analysis.json`.

**Two categories, kept explicitly separate, per instructions:**
- **Metadata evidence** (already known): 3/5 false positives' file paths
  reference two specific Facebook page accounts — a filename/path fact, not
  a pixel fact.
- **Visual evidence** (new, from actually viewing the images): false
  positives skew toward dense text-overlay + official/political photo
  subject matter + small recurring logo-sticker watermarks; false
  negatives skew toward classic Egyptian film-still meme templates with a
  plain caption bar and no watermark stickers, lower resolution/older film
  stock in some cases. Both are real, reproducible, pixel-level
  observations — not proof of a causal mechanism, but reported as
  hypotheses consistent with the evidence.

## 14. Phase 5B — text-only baseline, encoder comparison (complete)

Two frozen Arabic/multilingual text encoders were compared under identical
conditions (same leakage-clean split, same target, same head architecture,
same training/threshold procedure): `aubmindlab/bert-base-arabertv2`
(AraBERT, 135.2M params) and `microsoft/mdeberta-v3-base` (mDeBERTa,
278.2M params). **AraBERT was selected for Phase 5C** — it won on every
primary criterion (PR-AUC, F1, recall, FNR) and is the smaller, cheaper
model; not a tie, so the "prefer the simpler model" tiebreaker was not
needed. Full results: `docs/evaluation.md` §Phase 5B.

## 15. Phase 5C — multimodal fusion (complete)

Frozen CLIP (exact Phase 5A checkpoint) + frozen AraBERT (exact Phase 5B
checkpoint) embeddings concatenated (`[text_768 ; image_512]` = 1280-dim) →
a small trained fusion head.

**Precise result statement:** on the Prop2Hate-Meme evaluation set, frozen
CLIP + AraBERT late fusion improved F1 (0.638) and PR-AUC (0.628) over both
unimodal baselines (AraBERT F1 0.562/PR-AUC 0.612; CLIP F1 0.548/PR-AUC
0.565) at the fixed threshold 0.5, with the fusion ablation and 14 direct
complementarity cases providing evidence that image and text contribute
complementary signals. This was measured, not assumed — the hypothesis
("does combining representations capture complementary information neither
captures alone") was tested, not presumed true. **This is specific to this
dataset and task** — it is not a claim about multimodal fusion in general,
not a production-readiness claim, and not evidence of general multilingual
or internet-scale performance. Full results, ablation, and complementarity
analysis: `docs/evaluation.md` §Phase 5C.

**Missing-modality representation is an experimental ablation mechanism,
not a production fallback.** Ablation inputs use the **train-set mean
embedding vector** for the removed modality, not a zero vector — a zero
vector is out-of-distribution for a frozen encoder's output space and
would confound the ablation with an artificial input the model never saw
in training. This exists solely to test whether the fusion head uses both
modalities; it is not a real missing-image/missing-text handling strategy
for any production or serving path (none exists yet), and is not a learned
gating mechanism (that would require joint training, out of scope here).
See `src/safelens/models/multimodal/missing_modality.py`.

**No filename/account/metadata features were given to the model** — the
fusion pipeline only ever touches `.text` and the resolved image file
bytes; `image_path`/`example_id` are I/O plumbing to locate files, never
concatenated into the embedding (verified in code and by test).
