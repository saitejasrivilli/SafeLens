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
