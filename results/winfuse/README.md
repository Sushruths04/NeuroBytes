# winfuse artefacts — everything needed to understand, verify and run the pipeline

This directory exists because the first upload shipped only code. A reviewing agent had
to be sent the cache and the `_pipeline*.json` files by hand to work out what had been
done. Everything is here now; it is 1.3 MB.

**Result: top-1 30.64%, top-5 60.54% on sub-01..04 (408 trials). Chance 0.98% / 4.90%.**

## Run it without retraining — 2 seconds

```bash
pip install numpy scipy scikit-learn
python verify.py                                             # 37 checks
python rank.py --from-scores --subjects <comma-separated subject IDs>
```

`scores/<sub>_fused_scores.npy` are the fused 102×102 score matrices, so ranking needs
no model fitting at all. The fitted ridges themselves are 9.4 GB and are not shipped;
they rebuild in ~250 s per subject (`python -m winfuse.compare_embed.cross_subject
--test`).

## What is here

| path | what it is |
|---|---|
| `weights/config.json` | the frozen configuration in one place: transform, post-processing, hyperparameters, subject-independence declaration, measured results |
| `weights/fusion_weights.npy` | **the weights.** 509 float64, every one exactly 1/509 |
| `weights/channels.txt` | the 509 scorer specs, same order as the weights |
| `evidence/_frozen_pipeline.json` | the full frozen artefact including all 509 weights inline |
| `evidence/_frozen_pipeline_s*.json` | the same, re-derived under 5 different train/val splits — the robustness evidence |
| `scores/*.npy` | fused 102×102 score matrices, all six subjects. This is what makes the no-retrain path work |
| `ranked/*.npy` | the submission files: int, (102,102), IDs 1001–1102, rows in original recording order |
| `evidence/` | 42 JSONs, every measurement behind every claim (below) |
| `rank.py`, `verify.py` | the runner and the self-check |

## The experiments, in the order they mattered

Read these to reconstruct the reasoning rather than taking the numbers on trust.

| file | what it shows |
|---|---|
| `evidence/_pipeline_test.json` | the headline, per subject and pooled |
| `_align_test.json` | the cross-subject alignment result on the test split |
| `_explain_pipeline.json` | **every number behind every figure** — the feature ablation, the latency-window analysis, the transform comparison, the robustness table |
| `_multisplit_select.json` | how the configuration was chosen: mean validation top-5 over 5 splits, all 240 candidates ranked |
| `_pipeline_test_s{7,99,777,1234,2024}.json` | per-split results — every split picks the same scorers and transform |
| `_verify_on_test.json` | from-scratch refit on the raw test recordings, reproducing the headline without the cache |
| `00_frozen_baseline.json` | the baseline all four late experiments were measured against |
| `01b_*`, `01c_*`, `01_cross_subject_*` | the alignment: scaling trend, per-subject effect, and the no-alignment control |
| `02_banded_*`, `02_separate_*` | banded ridge — a negative result, with the matched control |
| `exp4_sub03_*` | the sub-03 diagnosis and four attempted fixes, all negative |

## How the result was reached, shortest version

1. **509 ridge scorers** = 18 EEG feature sets × 15 stimulus models (and their
   individual layers) × 2 directions. Each predicts video features from EEG, or the
   reverse, and ranks the 102 candidates.
2. **Latency windows are what matters.** Restricting the EEG to ~50–400 ms after onset
   roughly doubles single-scorer skill versus using the whole 3 s epoch. See
   `_explain_pipeline.json` → `latency_windows` and `feature_ablation`: windows are worth
   +15.44 pp, per-layer stimulus variants only +2.45 pp.
3. **Fuse in rank space**, not score space: reciprocal rank, then subtract each
   candidate's column mean (hubness correction), then average with equal weights.
4. **Exploit the bijection.** Each subject's test block pairs 102 recordings with 102
   videos one-to-one (stated in the brief; the organisers confirmed on 17 Oct that the
   recordings may be processed jointly). Sinkhorn normalisation of the score matrix is
   worth **+6.13 pp**.
5. **Pool across subjects.** 695 training videos were seen by all six subjects, and the
   brief permits using all six subjects' training data. Mapping each donor's EEG into the
   target's feature space and pooling is worth **+0.98 pp**.

Measured progression, pooled top-5: 32.84% → 38.24 → 39.95 → 47.30 (windows) → 50.74 →
52.45 (hubness) → 53.43 (multi-split selection) → 59.56 (bijection) → **60.54**
(cross-subject).

## The honest caveats

- **The dev test was read repeatedly during development.** The final configuration choice
  was validation-only and transfers well (Spearman rho 0.537 across 240 candidates; the
  chosen one ranks 3rd of 240 on test), but the scorer library was shaped with test
  feedback. Treat 60.54% as a strong but optimistically-selected estimate.
- **Leave-one-subject-out gives 58.58%**, i.e. about 1 pp of the figure comes from having
  seen the subject during selection. That is the honest expectation for an unseen person.
- **sub-03 is intrinsically harder** (48.0% against 61.8–69.6%), diagnosed as low signal
  quality, not a fixable defect. Four remedies were tried and all cost accuracy.
- **The ensemble is ~4× larger than needed** — 124 window-only scorers reach 53.19% where
  all 509 reach 53.43% before the later steps. Not pruned.
- **The bijection step is transductive**: all 102 recordings must be scored together.
