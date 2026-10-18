# Results

## Official development test (sub-01..04, 408 recordings, 102 candidates each)

| | top-1 | top-5 | mean rank |
|---|---|---|---|
| **NeuroBytes C13 (submission)** | **36.03%** | **67.16%** | **5.99** |
| C13 without Sinkhorn balancing (fallback) | 32.35% | 66.67% | 6.34 |
| Chance | 0.98% | 4.90% | 51.5 |

Per person (top-1): sub-01 49.0%, sub-02 32.4%, sub-03 26.5%, sub-04 36.3%.

With 408 recordings, one standard error of top-1 is about 2.4 points; differences smaller than that are within noise.

## How the score was built

| Step | top-1 | top-5 |
|---|---|---|
| Ridge baseline (one embedding) | 11.0% | 29.7% |
| winfuse: 509 latency-window scorers | 23.0% | 52.5% |
| + our 77 video-model scorers | 26.5% | 57.4% |
| + hubness correction (CSLS) | 30.9% | 62.3% |
| + cross-participant alignment, T2, Sinkhorn | 32.6% | 66.7% |
| + two contrastive EEG networks (C13) | 36.0% | 67.2% |

Every step was kept only if it improved cross-validation on held-out training videos.

## Robustness

| Check | Result |
|---|---|
| Retrain the networks with the same seeds on a fresh GPU machine | bit-identical scores for all six participants; C13 again 36.03 / 67.16 / 5.99 |
| Retrain with five new random seeds | 34.31 / 67.65 / 5.98 (top-1 −1.7, top-5 +0.5: within one standard error) |
| Validation on held-out training videos, all six participants | mean top-1 14.95%, top-5 39.67%, mean rank 16.85 (102-way, single recordings, no Sinkhorn) |
| Reproduce from released weights (clean Modal machine, source and Docker) | 36.03 / 67.16 / 6.00 (CPU); regenerated predictions identical to the submission in top-1 and top-5 |

Validation is lower than the development test because training recordings are single, noisier repetitions and are scored without test-set balancing.

## Selection protocol

All settings (latency windows, ridge penalties, fusion weights, network targets and size, CSLS/Sinkhorn parameters) were chosen by cross-validation on held-out training videos (20 random 102-way candidate blocks per split). The development-test labels of sub-01..04 were used to report scores and to compare the final candidates. They were not used to fit any model.

## Limitations

- **Short informative window.** Almost all decodable information lies 50–800 ms after video onset. Later parts of the 3-s recording carry little video-specific signal at 17 posterior channels.
- **Misses.** About half of the misses still have the correct video in the top five. Misses are driven by noise in individual recordings rather than by a fixed set of hard videos.
- **Differences between participants.** Top-1 ranges from 26% to 49% across the four development participants.
- **Batch post-processing.** CSLS and Sinkhorn use all test recordings of a participant jointly. A fallback without Sinkhorn is provided in `submissions/C13_no_sinkhorn/`.
- **Rebuild from raw data.** Rebuilding the winfuse scores from the raw data gives scores that correlate 0.987 with the shipped ones; in our tests this lowered C13's top-1 by about 3 points at unchanged top-5. The quickstart therefore uses the shipped winfuse scores. See [PIPELINE.md](PIPELINE.md).
