# Policy Taxonomy

## Status

Phase 2 (data pipeline) only. This document maps the **labels the current dataset actually
supports** to a moderation-relevant vocabulary — it does not yet define production
thresholds or actions (that's Phase 10, the decision engine).

## Available signal (from `google/civil_comments`)

The dataset provides 7 continuous scores per comment, each the fraction of annotators who
flagged that attribute, range `[0.0, 1.0]`:

| Dataset field       | Informal meaning                                   |
|----------------------|-----------------------------------------------------|
| `toxicity`           | general toxicity / rudeness                         |
| `severe_toxicity`    | severe/extreme toxicity                              |
| `obscene`            | obscene language                                     |
| `threat`             | threatening language                                 |
| `insult`             | insulting language                                   |
| `identity_attack`    | attacks based on identity (race, gender, etc.)       |
| `sexual_explicit`    | sexually explicit content                            |

These are **not** binary labels — a downstream classifier (Phase 3+) must pick and document
an explicit threshold (e.g. `>= 0.5`) to binarize them; the raw scores are preserved as-is
in the data contract so that decision is not made prematurely at ingestion time.

## Mapping to the target SafeLens taxonomy

The full target taxonomy (`CLAUDE.md` §10) is:
`SAFE, HARASSMENT, HATEFUL, VIOLENCE, SEXUAL, SELF_HARM, SPAM`.

Only the subset actually backed by this dataset is used for training/evaluation:

| SafeLens policy | Backed by dataset field(s)         | Coverage        |
|-----------------|--------------------------------------|-----------------|
| HARASSMENT      | `insult`, `toxicity`                 | Yes             |
| HATEFUL         | `identity_attack`                    | Yes             |
| VIOLENCE        | `threat`                             | Yes             |
| SEXUAL          | `sexual_explicit`, `obscene`         | Yes             |
| SAFE            | absence of the above (all scores low) | Yes (implicit) |
| SELF_HARM       | none                                  | **Not covered** |
| SPAM            | none                                  | **Not covered** |

**SELF_HARM and SPAM are not implemented** — `google/civil_comments` contains no signal for
either category, and no label is invented to fill the gap. If SafeLens later needs these
categories, a second dataset with that signal must be sourced and documented here before any
model is trained to predict them.

## Per-policy documentation (definition / ambiguity / action)

Full definition, examples, ambiguity discussion, and moderation action per policy will be
written once policy thresholds are studied in Phase 10 against held-out data (per
`CLAUDE.md` §20). Writing those decisions now, before any threshold analysis exists, would
be exactly the kind of unmeasured claim `CLAUDE.md` prohibits.
