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
- **Outcome:** established reference baseline for Phase 4 (DeBERTa) comparison. Not yet
  compared to Phase 4 — no transformer has been trained.
