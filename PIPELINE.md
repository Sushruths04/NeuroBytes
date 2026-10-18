# Reproducing C13 (native, without Docker)

C13 is our submission:
- **Dev test (sub-01..04):** top-1 36.03%, top-5 67.16%, mean rank 5.99.
- **Predictions:** `submissions/C13/`.

Docker is an optional shortcut (`docker/README.md`); everything below runs natively.

```
score = 0.85 * C10 blend + 0.075 * rrf+hub(network h6) + 0.075 * rrf+hub(network h10)  ->  CSLS k=2  ->  Sinkhorn(3, 0.5)
C10 blend = 0.85 * ((509/586) * winfuse aligned + (77/586) * ours77) + 0.15 * rrf+hub(T2)
```

## 0. Setup

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export DATA=/path/to/eeg-mind-reading-challenge   # official folder with trn/, tst/ (tst_stim/ for dev evaluation)
```

## 1. Stimulus features

The videos and extracted video frames are not shipped. Two options:

- **Download (recommended):** release `v1.0` contains:
  - our video/audio features (`outputs/mindread_features`);
  - the winfuse embedding banks (`cache/embeddings`).

  It is a 2.8 GB download and needs about 6 GB of free disk (archives plus extracted files). After extracting, you can delete the archives in `rebuild/`.

  `bash scripts/run_c13_pipeline.sh features`
- **Extract from the videos yourself:**
  - our features: `scripts/extract_features.py` and `scripts/modal_features_v2.py` (V-JEPA 2);
  - winfuse banks: `src/winfuse/embed_*` (see `results/winfuse/README.md`).

  Exact model names and revisions are in `MODELS.md`.

## 2. Ridge components (deterministic)

`bash scripts/run_c13_pipeline.sh scorers` runs:

- **winfuse aligned (509 scorers, ~63% of C13):** `python -m winfuse.compare_embed.cross_subject --test --subjects …`
  - For each target person, a ridge map from every donor person's EEG into the target's space is fitted on shared training videos.
  - Mapped donors are pooled; 509 ridge scorers are fitted and rank-fused.
  - About 1 h per person on 10 CPU cores.
  - Output is cached in the repo's `cache/align/` (never in the data folder, which may be read-only) and **reused** on the next run; set `FORCE=1` (or delete those files) to rebuild.
- **ours77 (~10%):** `python combo/ours77.py OUT sub-0X` (77 scorers on our features, ~1 min per person).
- **T2 (~11%):** `python combo/t2_scores.py OUT X` (time-resolved encoder; about 25 min for one person on a busy 20-core laptop CPU).

## 3. Networks (15%)

**Download** release `v1.0`, or **retrain:** `bash scripts/run_c13_pipeline.sh networks`.

- Two convolutional EEG encoders with six and ten target heads, five seeds each.
- **On a CUDA GPU:** about 2 minutes per network (RTX A2000), with bit-identical weights.
- **On a CPU:** about 4–5 minutes per seed, so roughly 45 minutes or more for all 10 seeds (there is no Apple-GPU path). CPU-retrained weights are similar but not bit-identical: in one test, swapping 4 CPU-retrained h6 seeds into the released set gave 35.05 / 67.16 / 6.05, within seed noise (new seeds: 34.31 / 67.65).

## 4. Assemble the submission

```bash
WEIGHTS=<c13o_weights_20261018 or rebuild/weights> bash scripts/run_c13_pipeline.sh assemble
```

Writes `submissions/C13_rebuilt/` and compares it with `submissions/C13/`.

- **Dev check:** `python combo/c11.py --dev` with the same environment variables (prints top-1/top-5 per person).
- **Post-processing:** CSLS k=2 and Sinkhorn use all 102 recordings of a person jointly. Confirm acceptance of all batch-dependent postprocessing with the organizers. A fallback without Sinkhorn is in `submissions/C13_no_sinkhorn/`.

## Verification

| Check | Result |
|---|---|
| Quickstart on a clean machine (fresh Python 3.11, `requirements.txt`) | 36.03 / 67.16 / 6.00; regenerated predictions identical to the submission in top-1 and top-5 |
| Docker image built from `docker/Dockerfile` on a clean machine | `evaluate` 36.03 / 67.16 / 6.00; `predict` identical top-1/top-5 |
| Retrain h6/h10 with the same seeds on a fresh GPU | bit-identical scores for all six participants |
| Retrain h6/h10 with new seeds 5–9 | 34.31 / 67.65 / 5.98 |
| 28 unit tests (`pytest tests`) | pass |

CPU and GPU arithmetic differ in the last digits, so deep ranks can reorder slightly; top-1 and top-5 are unaffected.

## Selection protocol

Every setting (latency windows, ridge penalties, fusion weights, network targets and size, CSLS/Sinkhorn parameters) was chosen by cross-validation on held-out training videos of all six participants (`combo.P.split_ids`, 20 random 102-way candidate blocks). The development-test labels of sub-01..04 were used to report scores and to compare the few final candidates; they were never used to fit a model.
