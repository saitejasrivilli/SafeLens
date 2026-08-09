# Model Design — Phase 3 Baseline

## Status

Classical ML baseline complete: TF-IDF + Logistic Regression on `google/civil_comments`
(Phase 2 processed split). Reference model for later transformer/multimodal comparisons.

## 1. Why TF-IDF

TF-IDF is the standard first baseline for text classification: no training-time GPU
requirement, fully interpretable (feature weights map directly to n-grams), fast to fit
(<1s on this dataset size), and gives a real floor to compare a transformer against later.
If DeBERTa in Phase 4 doesn't clearly beat this baseline, that's a signal worth reporting
(§46 of `CLAUDE.md`), not something to skip measuring.

## 2. Why Logistic Regression

Paired with TF-IDF, Logistic Regression is the standard combination: convex optimization
(deterministic given a fixed seed and solver), calibrated-enough probabilities for a
threshold sweep, and `class_weight="balanced"` handles the ~5% positive rate without
resampling. No large hyperparameter search was run — the goal is a transparent baseline,
not leaderboard optimization (per `CLAUDE.md` §3).

## 3. Why `toxicity` is the first target

`google/civil_comments` provides 7 continuous per-comment scores. `toxicity` is the
broadest, best-populated dimension (5.1% positive rate at the 0.5 ground-truth threshold,
vs. 0.09%–3.6% for the other six — see `docs/data_pipeline.md`), making it the only
dimension with enough positive examples in an 8,000-row prototype slice to fit and evaluate
a classifier meaningfully. The other six (`severe_toxicity`, `obscene`, `threat`, `insult`,
`identity_attack`, `sexual_explicit`) are deferred, not discarded — they represent more
specific policy-relevant sub-categories (e.g. `threat` → VIOLENCE, `identity_attack` →
HATEFUL; see `docs/policy_taxonomy.md`) that need either a larger sample or a different
class-imbalance strategy to train reliably. The architecture supports this: each dimension
is loaded independently via `load_split(split, target=<dimension>)`
(`src/safelens/models/text/data.py`), so adding a policy head per dimension later (Phase 6+
multi-label / multi-head models) does not require touching the data-loading or config
layers, only the model layer.

## 4. Continuous label → training label

`labels.<target>` is a continuous `[0,1]` score (fraction of annotators who flagged that
attribute) — not a ground-truth binary label. Two distinct thresholds exist and must not be
conflated:

- **`label.ground_truth_threshold`** (config: `configs/baseline.yaml`, value `0.5`) —
  converts the *dataset's* continuous annotator-agreement score into a binary label used to
  train and evaluate the classifier (`majority of annotators flagged this as toxic`).
- **`decision_threshold`** — operates on the *model's predicted probability*, and is
  selected via a validation sweep (§ below). This is what a production system would tune
  per deployment.

`ground_truth_threshold=0.5` was chosen as the conventional "majority agreement" cut and is
explicit in configuration precisely so it can be revisited — it was not chosen via any
data-driven search in Phase 3.

## 5. Threshold selection methodology

See `docs/evaluation.md` for the full threshold sweep and selection criterion. Summary: 9
candidate thresholds (0.1–0.9) were evaluated **on the validation set only**; the threshold
maximizing validation F1 was frozen (0.5) and used exactly once against the test set. The
test set was never used to pick or tune anything.

## 6. Class imbalance

Positive rate at the ground-truth threshold: train 5.12%, validation 5.10%, test 5.10%
(consistent across splits — expected, since the leakage-safe split is stratified on this
exact binarization). Handled via `class_weight="balanced"` in Logistic Regression, not
resampling — resampling was judged unnecessary complexity for a transparent baseline.

## 7. Architecture for future policy heads

`src/safelens/models/text/{config,data,pipeline,threshold,metrics,predict,artifacts}.py`
are all parameterized by `label.target` and don't assume `toxicity` is the only label. A
second policy head (e.g. `threat`) can reuse every module unchanged — only a second
`BaselineConfig` (or a shared multi-output model, deferred) is needed. The prediction output
(`src/safelens/models/text/predict.py::Prediction`) is deliberately policy-independent
(probability + threshold + predicted label, no ALLOW/REVIEW/BLOCK) so multiple heads can
feed into a single policy engine later (Phase 10).

## 8. Baseline limitations

- Trained on an 8,000-row prototype slice (Phase 2 scope), not the full 1.8M-row corpus.
- Single-label (`toxicity` only); the other 6 dimensions and SPAM/SELF_HARM are not modeled.
- TF-IDF cannot capture context/negation/sarcasm — expected to be the main source of the
  observed low recall on rare positive classes (see `docs/evaluation.md`).
- `ground_truth_threshold=0.5` is a convention, not derived from a labeling-cost analysis.

## 9. Leakage prevention

No new leakage risk introduced in Phase 3: the exact Phase 2 processed split
(`data/processed/civil_comments/{train,validation,test}.jsonl`) is loaded read-only via
`src/safelens/models/text/data.py::load_split`, which does not download, reshuffle, or
resplit anything. TF-IDF is fit on `train.texts` only; `validation`/`test` are only ever
`.transform()`-ed, never `.fit()`-ed on. Verified by `test_vectorizer_fits_only_on_training_vocabulary`.

## 10. What Phase 4 should improve

- Recall on the positive class (test recall 0.23 at the selected threshold — see
  `docs/evaluation.md`) is the most likely place a transformer (contextual embeddings)
  should show a real improvement over bag-of-n-grams.
- Consider whether a larger training slice (beyond 8,000 rows) is needed once GPU
  (Colab) training is available, given how few positive examples exist per rare dimension.
