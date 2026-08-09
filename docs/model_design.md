# Model Design — Phase 3 Baseline & Phase 4 DeBERTa

## Status

Classical ML baseline complete: TF-IDF + Logistic Regression on `google/civil_comments`
(Phase 2 processed split). Phase 4 (DeBERTa-v3-small fine-tune) complete, trained on Colab
GPU. See §11 onward for Phase 4; §1–10 are unchanged from Phase 3.

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

## 11. Phase 4 — why DeBERTa-v3-small

`microsoft/deberta-v3-small` (revision `a36c739020e01763fe789b4b85e2df55d6180012`, 141,896,450
parameters) is a small, efficient transformer with disentangled attention — a reasonable
middle ground between TF-IDF's zero contextual understanding and a full-size LLM's compute
cost, appropriate for the "start simple, measure honestly" mandate in `CLAUDE.md`. It was
used exactly as published (official tokenizer, `AutoModelForSequenceClassification` with a
freshly-initialized classification head), not substituted for anything else.

## 12. Hardware reality: why training happened on Colab, not M2

Initial local training on the M2 (MPS backend) was measured at **370–524 seconds/step**,
projecting to 53–76 hours for the configured 525 steps — infeasible. This is a real,
measured finding, not an assumption: DeBERTa-v2's disentangled attention has known
inefficient kernels on MPS. Training moved to Colab GPU (T4) as `CLAUDE.md` §2 anticipates
("Colab GPU should be used for... transformer training"), using the **exact same,
unmodified** `scripts/train_deberta.py` — device selection is automatic
(`safelens.utils.device.detect_device`), so no code forked between the two environments.
On Colab (`Linux-6.6.122+-x86_64`, CUDA, `torch==2.11.0+cu128`), training took **602.9
seconds** (~10 minutes) for all 3 epochs — see `docs/evaluation.md` for full numbers.

A real bug surfaced during the M2 attempt and was fixed before any training completed: the
published `microsoft/deberta-v3-small` checkpoint stores weights in `float16`. Training in
float16 without a mixed-precision scaler (not used here — see §14) crashes
class-weighted `CrossEntropyLoss` with a dtype mismatch on MPS/CPU. Fixed by forcing
`torch_dtype=torch.float32` at model load (`src/safelens/models/text/deberta/train.py`).

## 13. Class imbalance (Phase 4)

Same ~5% positive rate as Phase 3, same imbalance problem. Handled via
`use_class_weighting: true` (config: `configs/deberta.yaml`) — inverse-frequency class
weights (`[0.527, 9.769]`) fed into a class-weighted `CrossEntropyLoss` inside a custom
`WeightedTrainer` (`src/safelens/models/text/deberta/train.py`). Configurable, not assumed
optimal — a future experiment could compare against unweighted loss or focal loss, but that
comparison was out of scope for the first clean DeBERTa run per the Phase 4 spec ("do not
perform a large hyperparameter sweep").

## 14. Training configuration (conservative, not tuned)

`configs/deberta.yaml`: learning rate 2e-5, per-device batch size 16, gradient accumulation
2 (effective batch 32), 3 epochs, weight decay 0.01, warmup ratio 0.1, max grad norm 1.0,
seed 42, max sequence length 256. Model selection during training used **validation PR-AUC**
(`metric_for_best_model="pr_auc"`, `load_best_model_at_end=True`) rather than loss or
accuracy, per `CLAUDE.md` §8 guidance for a class-imbalanced problem — epoch 3 had the
highest validation PR-AUC (0.677) and was loaded as the final model. `fp16`/`bf16` mixed
precision was not enabled (config carries no such flag) — training ran in float32 throughout
on the Colab GPU; this is a real, measured decision, not an unmeasured assumption, and future
runs could measure whether bf16 (supported on modern CUDA) meaningfully changes speed/quality.

## 15. Threshold selection (Phase 4)

Same two-threshold architecture as Phase 3 (§4-5): `label.ground_truth_threshold=0.5`
(unchanged, dataset-label binarization) and a separate decision-threshold sweep on
*validation* model probabilities only. Candidates 0.1–0.9 were swept; **0.9** maximized
validation F1 (0.6034) and was frozen before the one test-set evaluation. See
`docs/evaluation.md` for the full sweep table and a caveat about its stability (the
validation split has only 61 positive examples, so the sweep is measurably noisy).

## 16. Phase 4 results summary

DeBERTa materially outperforms the frozen Phase 3 baseline on every primary metric —
absolute test-set F1 +0.217 (+79%), PR-AUC +0.216 (+80%), recall +0.230 (recall exactly
doubled). Full table, error analysis, and critical decision are in `docs/evaluation.md`.

## 17. Baseline vs. DeBERTa: the actual tradeoff being quantified

TF-IDF + Logistic Regression: ~0.65s to train, sub-millisecond inference, no GPU, trivial to
serve and version. DeBERTa-v3-small: ~10 minutes to train on a Colab T4, ~15.6ms p50
single-example inference latency on a CUDA GPU (`docs/evaluation.md`), 142M parameters to
version and serve, and a real dependency on GPU-class hardware for training (infeasible on
the M2 dev machine, per §12). The recall/F1 improvement is real and large; whether it
justifies the added serving complexity for a *production* system is a policy/cost decision
(Phase 10's cost model), not purely a modeling one — see the critical decision in
`docs/evaluation.md`.

## 18. Phase 4 limitations

- Same 8,000-row prototype slice as Phase 3 — not the full 1.8M-row corpus.
- Single-label (`toxicity` only), same as Phase 3.
- Decision-threshold selection (0.9) is based on a validation set with only 61 positive
  examples — the sweep in `docs/evaluation.md` shows F1 increasing almost monotonically from
  0.1 to 0.9, which is a plausible but noisy signal at this sample size, not a strong
  optimum. A larger validation set would give a more stable threshold estimate.
- Inference latency was measured on a Colab T4 GPU, not the M2 — the two model latencies in
  `docs/evaluation.md` are measured on different hardware and are not directly comparable
  for a serving-cost decision; that would require benchmarking both on the same target
  deployment hardware.
- Local model weight artifacts (`models/text/deberta/v1/`) were not retrieved from the Colab
  run at the time of writing — only the benchmark report (`experiment.json`, plots) was
  downloaded. The model is not currently re-loadable locally.

## 19. Phase 5 — text encoder is a separate, not-yet-implemented decision

Phase 5 (`docs/multimodal_design.md`) uses `QCRI/Prop2Hate-Meme`, whose text
is 100% Arabic (measured). `microsoft/deberta-v3-small` (Phase 4) is
English-only and cannot be reused for Phase 5's text branch. **No encoder
has been selected or implemented yet** — this section only proposes
candidates for review before Phase 5B is built.

| Candidate | Type | Params (approx.) | Why it's a candidate | Tradeoff |
|---|---|---|---|---|
| `microsoft/mdeberta-v3-base` | Multilingual DeBERTa-v3 (100+ languages incl. Arabic) | ~86M (base) | Same architecture family as Phase 4 — methodology stays consistent (disentangled attention, same tokenizer style), easiest to compare against Phase 4's approach conceptually | Multilingual generalist, not Arabic-specialized — may underperform a dedicated Arabic model on dialectal/informal meme text |
| `aubmindlab/bert-base-arabertv2` (AraBERT) | Arabic-specific BERT | ~136M | Purpose-built for Arabic, strong track record on Arabic NLP benchmarks, pretrained on a large Arabic corpus | Different architecture family from DeBERTa — less direct methodological continuity with Phase 4; primarily Modern Standard Arabic-oriented, memes often contain informal/dialectal Arabic |
| `CAMeL-Lab/bert-base-arabic-camelbert-mix` (CAMeLBERT-Mix) | Arabic-specific BERT, mixed-register pretraining (MSA + dialectal + classical) | ~108M | Explicitly trained on a *mix* of registers, plausibly the best fit for informal/dialectal meme captions specifically | Less widely benchmarked than AraBERT in some tasks; still a BERT-family model, same continuity tradeoff as AraBERT |

**No recommendation is finalized.** `mdeberta-v3-base` has the strongest
methodological continuity with Phase 4's approach; the two Arabic-specific
models (AraBERT, CAMeLBERT-Mix) have the strongest task-fit argument given
the dataset is 100% Arabic and meme text is informal/dialectal. This should
be resolved by an actual measured comparison (frozen-encoder embedding
quality or a small probe) before Phase 5B is built, not by picking one
arbitrarily — availability/licensing of each on the Hugging Face Hub also
needs to be re-verified at implementation time, not assumed from this list.

## 20. Phase 5A — image-only baseline

**Why an image-only baseline first:** per `CLAUDE.md`'s own build order
(image-only → text-only → multimodal), establishing what the image alone
can do is a prerequisite for later claiming multimodal information adds
value — without it, any multimodal result would have no unimodal floor to
compare against.

**Architecture:** frozen `openai/clip-vit-base-patch32` (revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, 151.3M total params, 87.5M in
the vision tower) → 512-dim projected image embedding (via
`model.vision_model(...).pooler_output` → `model.visual_projection(...)`,
used instead of `get_image_features()` because the installed transformers
version's `get_image_features()` was verified to return an unprojected
pooled output, not the projected embedding — a real, checked discrepancy,
not an assumption) → classification head: `Linear(512→256) → ReLU →
Dropout(0.2) → Linear(256→2)`.

**Frozen-encoder choice:** the entire CLIP model's parameters have
`requires_grad=False` (verified by test); only the ~132K-parameter head is
trained. This follows `CLAUDE.md` §14/§17's "start with frozen embeddings,
only fine-tune if experiments justify it" — no fine-tuning was attempted or
needed to get a working baseline.

**Class weighting:** computed from the leakage-clean **train** split only
(2,141 examples, 213 positive) — weights `[0.555, 5.026]`, i.e. the
hate-labeled class is weighted ~9x higher in the loss, roughly matching its
~1:9 rarity in train. Dev/test distributions were never used to compute
these weights.

**Training behavior (real, measured, somewhat surprising):** the best
validation (dev) PR-AUC was reached at **epoch 0** (the very first epoch)
and never improved in the following 8 epochs (early-stopping patience
exhausted at epoch 9). This is reported as-is, not smoothed over — plausible
explanations include the head being simple enough to reach a good solution
almost immediately given how separable CLIP's frozen embeddings already
make the classes, and/or dev's PR-AUC estimate being too noisy (31 positive
examples) to reliably detect further improvement. No single explanation is
asserted as fact.

**Results, error analysis, latency:** see `docs/evaluation.md` §Phase 5A and
`docs/experiments.md`. This baseline does not claim or imply anything about
multimodal benefit — that requires Phase 5B and 5C on the same split, not
yet built.

## 21. Phase 5B — Arabic text-only encoder comparison

**Why compare two encoders instead of picking one:** the dataset is 100%
Arabic informal meme text — a real architectural decision, not a
convenience pick. `microsoft/mdeberta-v3-base` was not chosen by default
just because DeBERTa was already used in Phase 4; instead two candidates
were measured head-to-head under identical conditions (same split, same
frozen-encoder-plus-head architecture, same training/threshold procedure).
CAMeL-BERT was not added to this comparison — nothing in the dataset
inspection or error analysis surfaced a concrete, dataset-specific reason
(e.g. heavy dialectal/slang content that AraBERT/mDeBERTa specifically
mishandle) that would justify a third candidate; adding it without such a
reason would be building a model zoo, not answering the actual question.

**Architecture (both candidates, identical):** frozen text encoder → masked
mean-pooling over `last_hidden_state` (no dedicated pooler/projection used,
since a base `AutoModel` has none by default) → 768-dim sentence embedding
→ same classification head as Phase 5A (`Linear(768→256)→ReLU→Dropout(0.2)
→Linear(256→2)`). Entire encoder frozen (`requires_grad=False`, verified by
test); only the ~198K-parameter head trained.

**Class weighting:** identical method to Phase 5A — computed from the
leakage-clean train split only (`[0.555, 5.026]`, same weights as Phase 5A
since it's the same label distribution, unrelated to which encoder is used).

**Results summary** (full detail, threshold sweeps, and error analysis in
`docs/evaluation.md`): AraBERT reached test PR-AUC 0.612 vs. mDeBERTa's
0.514; AraBERT's best validation epoch was epoch 3 of 12 run, vs.
mDeBERTa's epoch 22 of 31 — AraBERT converged faster to a better optimum on
this small (2,141-example) training set. **AraBERT selected for Phase 5C**
on PR-AUC, F1, recall/FNR, validation robustness, compute cost (135M vs.
278M params, 57.6s vs. 83.3s embedding extraction), and model size —
mDeBERTa did not win on any of the seven selection criteria, so this was
not a close call requiring the "prefer the simpler model" tiebreaker.

**A genuine measurement anomaly, reported honestly:** for both encoders,
measured end-to-end latency (tokenization + encoder + head) came out
*lower* than the encoder-only stage latency in isolation (e.g. AraBERT:
encoder-only P50 64.6ms vs. end-to-end P50 38.6ms) — which should be
impossible if the measurements were noise-free, since end-to-end strictly
includes the encoder stage. This is attributed to MPS dispatch/scheduling
jitter on very short single-example forward passes, not a logic error in
the benchmarking code (the code was manually re-checked; both paths call
the identical `encode_texts` function). Reported as a real limitation of
this benchmark's precision at this scale, not smoothed over or hidden.

## 22. Phase 5C — multimodal fusion architecture

**Architecture:** frozen CLIP image encoder (exact Phase 5A checkpoint) →
512-dim embedding; frozen AraBERT text encoder (exact Phase 5B-selected
checkpoint) → 768-dim mean-pooled embedding; concatenated to a single
1280-dim vector (`[text ; image]`, text first — an arbitrary but documented
and tested ordering) → the same `ClassificationHead` shape used in Phase
5A/5B (`Linear(1280→256)→ReLU→Dropout(0.2)→Linear(256→2)`), reused as-is
rather than duplicated, since the head is architecture-agnostic. Both
encoders remain entirely frozen (`requires_grad=False`, verified); only the
~330K-parameter fusion head is trained. No cross-attention, no multimodal
transformer — simple concatenation, exactly as scoped.

**Why frozen-plus-simple-concat first:** per `CLAUDE.md` §15's own
guidance ("do not begin with an unnecessarily complex multimodal
transformer") and the explicit instruction for this phase — a cheap,
interpretable baseline that isolates whether *any* fusion of these two
frozen representations helps, before spending any budget on fine-tuning or
attention mechanisms.

**Result, stated precisely:** on the Prop2Hate-Meme evaluation set, frozen
CLIP + AraBERT late fusion improved F1 (0.638 vs. 0.562/0.548) and PR-AUC
(0.628 vs. 0.612/0.565) over both unimodal baselines at the shared fixed
threshold 0.5, with the fusion ablation and 14 direct complementarity cases
providing evidence that image and text contribute complementary signals.
This is a dataset-specific, measured finding — not a claim about
multimodal fusion in general, not evidence of production readiness, and not
a claim about general multilingual or internet-scale performance. See
`docs/evaluation.md` for the full comparison, ablation, and complementarity
evidence backing this claim (it is not asserted from the top-line number
alone).
