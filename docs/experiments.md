# Experiments

## Experiment 001 — TF-IDF + Logistic Regression baseline (Phase 3)

- **Date:** 2026-08-09
- **Dataset version:** `google/civil_comments@train:0-8000` (Phase 2 manifest:
  `data/manifests/civil_comments_manifest.json`, `processed_data_hash`
  `0db4bc377b5839603395f36afc8a47f016c4c5bc3ebd01f4cb10dddb8a39ac09`)
- **Split:** train 5,588 / validation 1,197 / test 1,197 (leakage-safe, Phase 2)
- **Config:** `configs/baseline.yaml` — TF-IDF (`ngram_range=(1,2)`, `min_df=2`,
  `max_df=0.95`, `max_features=20000`, `sublinear_tf=true`); Logistic Regression
  (`C=1.0`, `class_weight=balanced`, `max_iter=1000`, `random_state=42`)
- **Target label:** `toxicity`, binarized at `ground_truth_threshold=0.5`
- **Decision threshold:** swept 0.1–0.9 on validation, selected `0.5` by max validation F1
  (0.362)
- **Result:** test F1 0.275, precision 0.341, recall 0.230, PR-AUC 0.271, ROC-AUC 0.772,
  FPR 0.024, FNR 0.770 — full detail in `docs/evaluation.md` and
  `benchmarks/results/baseline/experiment.json`
- **Artifacts:** `models/baseline/v1/` (vectorizer, model, config, metadata — gitignored,
  not committed; reproducible via `make baseline-train`)
- **Reproducibility:** ran twice, bit-identical results excluding wall-clock timing fields
- **Outcome:** established reference baseline for Phase 4 (DeBERTa) comparison.

## Experiment 002 — DeBERTa-v3-small fine-tune (Phase 4)

- **Date:** 2026-08-09
- **Dataset version:** identical to Experiment 001 (same manifest, same hash, same split —
  no redownload, no reshuffle)
- **Model:** `microsoft/deberta-v3-small`, revision `a36c739020e01763fe789b4b85e2df55d6180012`,
  141,896,450 parameters, max sequence length 256
- **Training environment:** Google Colab GPU (Linux, CUDA T4, `torch==2.11.0+cu128`,
  `transformers==5.13.1`) — **not** the M2 dev machine. Local M2 (MPS) training was
  attempted first and measured at 370–524 seconds/step (53–76h projected), infeasible; see
  `docs/model_design.md` §12. Same unmodified `scripts/train_deberta.py` used in both
  attempts; only the execution environment differs.
- **Config:** `configs/deberta.yaml` — lr `2e-5`, batch 16 (effective 32 w/ grad
  accumulation 2), 3 epochs, weight decay 0.01, warmup ratio 0.1, seed 42, class-weighted
  loss (`[0.527, 9.769]`), model selection on validation PR-AUC
- **Target label:** `toxicity`, binarized at `ground_truth_threshold=0.5` (same as Exp. 001)
- **Decision threshold:** swept 0.1–0.9 on validation, selected `0.9` by max validation F1
  (0.603) — noted as a noisy estimate (validation set has only 61 positive examples)
- **Training time:** 602.9 seconds (~10 min), 3 epochs
- **Result (test, threshold=0.9):** F1 0.491, precision 0.528, recall 0.459, PR-AUC 0.487,
  ROC-AUC 0.932, FPR 0.022, FNR 0.541 — full detail in `docs/evaluation.md` and
  `benchmarks/results/deberta/experiment.json`
- **vs. Experiment 001:** F1 +0.217 (+79%), PR-AUC +0.216 (+80%), recall +0.230 (doubled),
  FNR -0.230 (-30%) — every primary metric improved
- **Artifacts:** `models/text/deberta/v1/` (gitignored; not yet populated locally — only
  the benchmark report was retrieved from Colab, not the model weights zip)
- **Reproducibility:** single run; not repeated a second time in this phase (unlike
  Experiment 001's two-run verification) — documented as a limitation
- **Bug found and fixed during this experiment:** the published checkpoint loads in
  `float16`, which crashed class-weighted `CrossEntropyLoss` on MPS/CPU with a dtype
  mismatch; fixed by forcing `torch_dtype=torch.float32` at load time. Also fixed: the
  inference-benchmark report hardcoded `"Measured locally on Apple M2"` regardless of actual
  training hardware — corrected to report the real device/platform dynamically.
- **Outcome:** DeBERTa shows a large, consistent quality improvement over the frozen
  baseline. See `docs/evaluation.md` for the full critical-decision writeup — conclusion is
  "yes, moderately-to-strongly on quality, with serving-cost tradeoffs not yet fully
  quantified," not an unqualified "ship it."

## Experiment 003 — Phase 5A: frozen-CLIP image-only baseline (Prop2Hate-Meme)

- **Date:** 2026-08-09
- **Dataset:** `QCRI/Prop2Hate-Meme` (CC-BY-NC-SA-4.0) — a **separate, not-combined** track
  from Experiments 001-002's `civil_comments`. Leakage-clean processed split: train 2,141 /
  dev 312 / test 606 (see `docs/multimodal_design.md` §10 for the remediation that produced
  the 2,141 train count from the official 2,143).
- **Model:** frozen `openai/clip-vit-base-patch32` (revision
  `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, 151.3M params, entirely frozen) + trained
  classification head (`Linear(512→256)→ReLU→Dropout(0.2)→Linear(256→2)`, ~132K trainable
  params). **No text used anywhere** (enforced and tested).
- **Config:** `configs/image_baseline.yaml` — lr `1e-3`, batch 64, up to 50 epochs (early
  stopping patience 8), weight decay 0.01, seed 42, class-weighted loss (`[0.555, 5.026]`,
  computed from train only), model selection on dev PR-AUC.
- **Target label:** `hate_label` (binary).
- **Decision threshold:** swept 0.1-0.9 on dev only, selected `0.6` by max dev F1 (0.356).
- **Training:** best epoch was epoch 0 (first epoch), 9 epochs run before early stopping —
  reported as-is, not explained away (see `docs/model_design.md` §20 and
  `docs/evaluation.md` for the two plausible-but-unconfirmed reasons). Embedding extraction
  87.1s, head training 0.66s, both on Apple M2 (MPS).
- **Result (test, threshold=0.6):** F1 0.531, precision 0.615, recall 0.468, PR-AUC 0.565,
  ROC-AUC 0.758, FPR 0.100, FNR 0.532. Result at fixed threshold 0.5 (F1 0.548, recall 0.701)
  also reported separately — full detail in `docs/evaluation.md` and
  `benchmarks/results/multimodal/image_only/experiment.json`.
- **Distribution shift:** test positive rate (25.41%) is 2.56x train/dev's (~9.94%),
  measured and kept visible in every reported metric, not corrected for.
- **Artifacts:** `models/vision/clip/v1/` (head weights only, 520KB, gitignored; CLIP
  weights never saved, per instructions).
- **Real issue found and fixed during this experiment:** `CLIPModel.get_image_features()`
  in the installed transformers version was verified to return an unprojected pooled output,
  not CLIP's published 512-dim projected embedding — worked around by calling
  `vision_model(...).pooler_output` → `visual_projection(...)` explicitly, verified against
  the expected 512-dim output shape.
- **Outcome:** establishes the image-only floor. **Makes no claim about multimodal benefit**
  — that comparison requires Phase 5B (text-only, not yet built) and Phase 5C (fusion, not
  yet built) on this same split.
- **Follow-up visual error analysis (same experiment, added later):** manual pixel-level
  review of the same 5 FP/5 FN found a real visual pattern (dense text overlay + official
  photo + watermark stickers for FPs; plain-caption film-template with no watermark for FNs)
  -- kept explicitly separate from the earlier filename/account metadata finding. See
  `benchmarks/results/multimodal/image_only/visual_error_analysis.json` and
  `docs/evaluation.md`.

## Experiment 004 — Phase 5B: Arabic text-only encoder comparison (AraBERT vs. mDeBERTa)

- **Date:** 2026-08-09
- **Dataset:** same leakage-clean Prop2Hate-Meme split as Experiment 003 (train 2,141 / dev
  312 / test 606). No image or filename information used (enforced, tested).
- **Candidates:** frozen `aubmindlab/bert-base-arabertv2` (rev `97522efce...`, 135.2M params)
  and frozen `microsoft/mdeberta-v3-base` (rev `a0484667b2...`, 278.2M params). CAMeL-BERT
  deliberately not added -- no dataset-specific reason surfaced to justify a third candidate.
- **Architecture (both):** frozen encoder -> masked mean-pooling -> 768-dim embedding ->
  same classification head as Experiment 003 (`Linear(768->256)->ReLU->Dropout(0.2)
  ->Linear(256->2)`), ~198K trainable params either way.
- **Config:** identical training/threshold procedure for both -- lr 1e-3, batch 64, up to 50
  epochs (patience 8), weight decay 0.01, seed 42, class weights `[0.555, 5.026]` (from
  train only), model selection on dev PR-AUC, threshold swept 0.1-0.9 on dev only.
- **Result (test, threshold=0.5):** AraBERT F1 0.562 / PR-AUC 0.612 / recall 0.805; mDeBERTa
  F1 0.525 / PR-AUC 0.514 / recall 0.604.
- **Result (test, dev-selected threshold):** AraBERT @0.8 F1 0.504 / PR-AUC 0.612 / recall
  0.409 / FNR 0.591; mDeBERTa @0.7 F1 0.359 / PR-AUC 0.514 / recall 0.253 / FNR 0.747.
- **Training:** AraBERT best epoch 3/12 run; mDeBERTa best epoch 22/31 run -- AraBERT
  converged faster to a better optimum. Embedding extraction: AraBERT 57.6s, mDeBERTa 83.3s
  (both M2 MPS).
- **Selected encoder: AraBERT** -- won on every criterion (PR-AUC, F1, recall/FNR,
  validation robustness, compute cost, model size); not a close call.
- **Real issue found and reported, not hidden:** both models' measured end-to-end latency
  came out lower than their encoder-only latency in isolation -- physically inconsistent if
  noise-free, attributed to MPS scheduling jitter on very short forward passes; code path
  verified correct, numbers reported as directionally informative only.
- **Error analysis (AraBERT):** high-confidence false positives cluster on gender-relations
  discourse (including text that reads as neutral or explicitly fairness-arguing) --
  possible topic-word over-association, not sentiment-based. High-confidence false
  negatives are casual slang/mockery with no overt hateful vocabulary -- consistent with
  Phase 4's finding that implicit, context-dependent hatefulness is hard to catch without
  explicit lexical signal.
- **Outcome:** AraBERT selected for Phase 5C. Neither text model nor Phase 5A's image model
  is claimed superior in isolation -- the three-way comparison table
  (`docs/evaluation.md`) is reported as-is; text (AraBERT) currently leads on PR-AUC, image
  leads on recall/FNR/F1 at their respective selected thresholds.

## Experiment 005 — Phase 5C: frozen CLIP + frozen AraBERT fusion

- **Date:** 2026-08-09
- **Dataset:** same leakage-clean split as Experiments 003-004 (train 2,141 / dev 312 /
  test 606). No filename/account/metadata features given to the model (enforced, tested).
- **Architecture:** exact Phase 5A CLIP checkpoint (frozen) + exact Phase 5B-selected
  AraBERT checkpoint (frozen) -> concatenate [text_768 ; image_512] -> fusion head
  (`Linear(1280->256)->ReLU->Dropout(0.2)->Linear(256->2)`, ~330K trainable params, same
  head class reused from Phase 5A/5B).
- **Missing-modality ablation representation:** train-set mean embedding per modality, NOT
  a zero vector -- documented design decision (out-of-distribution concern), not a learned
  mechanism.
- **Result (test, threshold=0.5, PRIMARY):** F1 0.638, PR-AUC 0.628, precision 0.572,
  recall 0.721, FPR 0.184, FNR 0.279. On the Prop2Hate-Meme evaluation set, frozen CLIP +
  AraBERT late fusion improved F1 and PR-AUC over both unimodal baselines (CLIP F1
  0.548/PR-AUC 0.565; AraBERT F1 0.562/PR-AUC 0.612) -- not generalized beyond this
  dataset/task.
- **Ablation:** fusion with one input replaced by train-mean drops to ~0.54 F1 either way --
  the fusion advantage requires both real inputs, not just architecture.
- **Complementarity (mandatory, full test-set scan):** 14 examples where both standalone
  models were wrong and fusion was correct (direct complementarity evidence); 24+25 examples
  where fusion lost information one standalone model had (a real, reported trade-off, not
  hidden). Multimodal standalone accuracy 79.2% vs. text-only 68.2% / image-only 70.6%.
- **Threshold sweep (dev-only, secondary):** selected 0.8; reported separately from the
  primary 0.5 comparison per instructions, not used to override it.
- **Training:** best epoch 3/12 run, 1.17s head training, 207.1s embedding extraction (both
  encoders, all 3 splits) on M2 MPS.
- **Latency (M2, same run):** end-to-end P50 66.4ms -- consistent with summing CLIP (36.7ms)
  + AraBERT (28.5ms) component latencies, no anomaly this time (unlike Experiment 004).
- **Success criteria outcome: B (moderate improvement)** -- real, measured, not "strong"
  (AraBERT alone still wins recall/FNR) and not "marginal" (a real, complementarity-backed
  F1/PR-AUC gain over both standalones). Not tuned to force a positive result -- the
  ablation and complementarity analysis were run specifically to test, not assume, the
  hypothesis.
- **Outcome:** first Phase 5 evidence that combining modalities adds real value for this
  dataset/task, with an honestly reported trade-off (fusion sometimes loses information a
  standalone model had). Not claimed production-ready.
