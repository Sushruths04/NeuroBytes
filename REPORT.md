# NeuroBytes: technical report

**Team BierBytes (Sushruth Shivaraju, Ayush Shetty). SchubKI National AI Hackathon 2026, EEG mind-reading challenge.**

NeuroBytes ranks 102 candidate videos from a 3-second EEG recording so that the video the person watched comes first. The final system, **C13**, reaches **36.03% top-1, 67.16% top-5 and a mean rank of 5.99** on the official development test (sub-01..04, 408 recordings; chance 0.98% / 4.90% / 51.5).

This report describes the data handling, every component of the final pipeline, the libraries used and the analyses that led to the final design. Commands to reproduce each step are in [PIPELINE.md](PIPELINE.md); exact model versions are in [MODELS.md](MODELS.md).

---

## 1. Task and data

- **Input:** one EEG recording per watched clip, 17 posterior channels (Pz P3 P7 O1 Oz O2 P4 P8 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2), 3 s at 500 Hz (17 × 1500 samples).
- **Output:** for each recording, a ranking of the 102 test videos (IDs 1001–1102). Submission files are 102 × 102 integer arrays in the original recording order (`submissions/C13/`).
- **Data used:** only the official challenge data: the EEG recordings of six participants (850–970 training clips each), the stimulus videos, and the development-test labels of sub-01..04.
- **Organizer preprocessing (as provided):** band-pass 0.1–40 Hz, blink components removed, downsampled from 1000 to 500 Hz, 3-s epochs. Test recordings are averaged over repeated viewings.

## 2. Our preprocessing

All components start from the same EEG preprocessing (`src/winfuse/common/preprocess.py`):

| Step | Setting |
|---|---|
| Band-pass | 0.5–40 Hz, Butterworth order 4, zero-phase (`scipy.signal.sosfiltfilt`) |
| Notch | 50 Hz, Q = 30 (`iirnotch` + `filtfilt`) |
| Downsampling | 500 → 250 Hz (factor 2) |
| Normalisation | per-channel z-score, statistics from the participant's training recordings |

On top of this, each component selects the part of the recording it uses:

- **Latency windows** for the ridge scorers: 50–200, 50–250, 50–300, 50–400, 50–500, 50–600, 75–275, 80–320, 100–300, 100–400, 120–450, 150–350, 200–400, 200–500 and 400–700 ms after clip onset, plus ERP-window, raw-signal and band-power feature sets (18 EEG feature sets in total).
- **Networks:** the first 600 ms (150 samples at 250 Hz), each recording scaled by its standard deviation.
- **T2 encoder:** seven 50-ms windows from 50 to 400 ms, at 100 Hz.

## 3. Video features

Every scorer compares the EEG with features of the candidate videos, computed once with frozen pretrained models (no fine-tuning):

- **Image and video models:** AlexNet and ResNet-50 (individual layers), CLIP ViT-B/32 and ViT-L/14, DINOv2 (small and base, several layers), SigLIP-B/16, VideoMAE-B, X-CLIP, V-JEPA 2 (giant layer 30; large layers 12/18/24), S3D (Kinetics-400).
- **Audio models:** CLAP, Whisper, wav2vec 2.0, AST.
- **Hand-crafted:** luminance, contrast, motion and audio-envelope descriptors (OpenCV, librosa).

Names and revisions of all models: [MODELS.md](MODELS.md).

## 4. The final pipeline (C13)

```
score     = 0.85 × C10 blend + 0.075 × rrf+hub(network h6) + 0.075 × rrf+hub(network h10)
            → CSLS (k = 2) → Sinkhorn (3 iterations, τ = 0.5) → ranking
C10 blend = 0.85 × ((509/586) × winfuse + (77/586) × ours77) + 0.15 × rrf+hub(T2)
```

C13 combines about 590 brain-to-video scorers from four families.

### 4.1 winfuse: 509 latency-window ridge scorers

- **Scorers:** 18 EEG feature sets × 15 stimulus models (and their layers), in two directions: 491 decoders (EEG → video features) and 18 encoders (video features → EEG).
- **Model:** PCA of the EEG features to 150 dimensions and of the video embeddings to 256 dimensions, then ridge regression with the penalty chosen by 5-fold cross-validation over 15 values from 10⁻¹ to 10⁶. Each scorer scores the 102 candidates by similarity between predicted and actual features.
- **Cross-participant alignment:** for each target participant, a ridge map is fitted from every other participant's EEG into the target's channel space, using the 695 training videos all six participants watched. The mapped recordings are added to the target's training data.
- **Fusion:** reciprocal-rank fusion (k = 60), hubness correction (subtract each candidate's mean score), equal weights (1/509 each).

### 4.2 ours77: 77 feature scorers

The same 15 latency windows decoded onto V-JEPA 2 giant, V-JEPA 2 large, S3D, CLAP and SigLIP features, plus 2 encoders (`combo/ours77.py`), fused with the same rank fusion and hubness correction.

### 4.3 T2: time-resolved encoding model

An anchor scorer plus 35 stream × window encoder/decoder scorers (`combo/t2_scores.py`, `configs/frozen_T2.json`):

- **Streams:** image (SigLIP, DINOv2-B), V-JEPA 2 giant L30, S3D, CLAP audio and low-level features.
- **Windows:** the seven 50-ms windows from 50 to 400 ms.
- **Weights:** non-negative, L2 = 1.0, learned per participant on calibration splits of their own training videos (mean over 3 seeds), then refit on all training videos.

### 4.4 Two contrastive EEG networks (h6, h10)

A convolutional EEG encoder trained on the training recordings of all six participants together (`combo/neural_clip.py`):

| Part | Setting |
|---|---|
| Spatial layer | one learned 17 → 32 channel mix per participant |
| Temporal trunk | 3 × Conv1d (64 channels, kernels 9/9/7, strides 1/2/2), BatchNorm, GELU, dropout 0.3 |
| Embedding | flatten → dropout 0.5 → linear 256 → GELU |
| Heads | one linear projection per target embedding, L2-normalised |
| Loss | symmetric InfoNCE against the batch, learnable temperature |
| Training | AdamW (lr 10⁻³, weight decay 0.05), one-cycle schedule, batch 256, 40 epochs |
| Augmentation | ±20 ms time shift, Gaussian noise (σ = 0.1), 10% channel dropout |
| Ensemble | 5 seeds per network |

- **h6** has six target heads: CLIP ViT-L/14, V-JEPA 2 giant L30, DINOv2-B (layers 4/8/12), AlexNet fc6, ResNet-50 layer 3, VideoMAE-B.
- **h10** adds four heads: AlexNet conv5, Whisper, hand-crafted visual features, X-CLIP video.

Each network scores the candidates by cosine similarity per head. Its scores enter C13 after rank fusion and hubness correction.

### 4.5 Post-processing

- **CSLS (k = 2):** removes "hub" videos that score highly against every recording, using each person's 102 test recordings together.
- **Sinkhorn (3 iterations, τ = 0.5):** row z-score, exponentiate, then alternately normalise rows and columns, so that each video is matched about once across the 102 recordings.

The same pipeline without Sinkhorn is in `submissions/C13_no_sinkhorn/`.

### 4.6 Model selection

All settings were chosen by cross-validation on held-out training videos of all six participants (20 random 102-way candidate blocks per split). These include the latency windows, ridge penalties, fusion weights, network targets and size, and the CSLS/Sinkhorn parameters. Every step was kept only if it improved this validation.

## 5. Analyses during development

### 5.1 Where in time the information is

We decoded from one latency window at a time and measured the mean rank of the true video (of 102; chance 51.5):

| Window (ms) | 50–200 | 50–400 | 200–500 | 480–830 | 910–1260 | 1340–1690 | 1760–2110 | 2190–2540 | 2620–2970 |
|---|---|---|---|---|---|---|---|---|---|
| Mean rank | 35.5 | **27.1** | 32.9 | 45.9 | 48.3 | 52.2 | 51.2 | 51.0 | 50.1 |

The information is in the early visual response, 50–400 ms after onset: P1 (about 80–130 ms), N1 (about 120–200 ms) and object-category information at 130–170 ms. Later windows are near chance. Restricting the EEG to latency windows was the largest single gain in winfuse: pooled top-5 rose from 39.95% to 47.30%.

### 5.2 Which video features match the EEG

The best features were mid-level CNN layers: ResNet-50 layer 3 (validation MRR 0.2016) and AlexNet fc6 (0.1984). They ranked ahead of the pooled outputs of DINOv2, CLIP and VideoMAE. Treating each layer as its own representation added 2.45 points. This matches the known correspondence between network depth and response latency in MEG and EEG.

### 5.3 Fusion and post-processing

- **Rank fusion:** combining scorers by reciprocal rank lets 509 scorers with different scales be averaged without calibration.
- **Hubness correction:** added 1.7 points of top-5 in winfuse. In C13, CSLS added 4.4 points of top-1 (26.5% → 30.9%).
- **Sinkhorn balancing:** raised winfuse top-5 from 53.43% to 59.56%.

### 5.4 Learning across participants

Aligned pooling of the other participants' recordings added 0.98 points of top-5 on the development test, and every participant improved. Pooling the same recordings without alignment lowered validation by 7.05 points. The alignment, not the extra data, carries the gain.

### 5.5 How the score was built

| Step | Top-1 | Top-5 |
|---|---|---|
| Ridge baseline (one embedding) | 11.0% | 29.7% |
| winfuse: 509 latency-window scorers | 23.0% | 52.5% |
| + 77 video-model scorers (ours77) | 26.5% | 57.4% |
| + hubness correction (CSLS) | 30.9% | 62.3% |
| + cross-participant alignment, T2, Sinkhorn (C10) | 32.6% | 66.7% |
| + contrastive network (C11) | 34.6% | 66.2% |
| + 7-head contrastive network (C12) | 35.0% | 67.4% |
| + two contrastive networks h6, h10 (**C13**) | **36.0%** | **67.2%** |

## 6. Results

### 6.1 Development test (sub-01..04, 408 recordings)

| | Top-1 | Top-5 | Mean rank |
|---|---|---|---|
| **C13** | **36.03%** | **67.16%** | **5.99** |
| C13 without Sinkhorn | 32.35% | 66.67% | 6.34 |
| Chance | 0.98% | 4.90% | 51.5 |

| Participant | Top-1 | Top-5 |
|---|---|---|
| sub-01 | 49.0% | 73.5% |
| sub-02 | 32.4% | 68.6% |
| sub-03 | 26.5% | 57.8% |
| sub-04 | 36.3% | 68.6% |

Where the true video lands: rank 1 in 36.0% of recordings, rank 2 in 10.3%, ranks 3–5 in 20.8%, 6–10 in 16.9%, 11–20 in 10.0% and beyond 20 in 5.9%.

### 6.2 Robustness

| Check | Result |
|---|---|
| Retrain the networks with the same seeds on a fresh GPU machine | bit-identical scores for all six participants |
| Retrain the networks with five new seeds | 34.31% / 67.65% / 5.98 |
| Cross-validation on held-out training videos, all six participants | 14.95% / 39.67% / 16.85 (single recordings, no Sinkhorn) |
| Reproduce from the released weights on a clean machine, natively and in Docker | 36.03% / 67.16%; regenerated predictions identical to the submission in top-1 and top-5 |

### 6.3 Comparison

| System | Task | Top-1 | Top-5 |
|---|---|---|---|
| **C13** | 102-way, 3-s video, 17-channel EEG | **36.03%** | **67.16%** |
| winfuse alone | 102-way, same data | 30.6% | 60.5% |
| Organizers' reference baseline | 102-way, same data | 1.7% | 9.3% |
| NICE (Song et al., ICLR 2024) | 200-way zero-shot, THINGS-EEG2 images | 15.6% | 42.8% |

## 7. Software

| Area | Libraries (versions in `requirements.txt`) |
|---|---|
| Core | Python 3.11, NumPy 2.0.2, SciPy 1.17.1, scikit-learn 1.9.1, joblib 1.6.0 |
| Networks | PyTorch 2.4.1, torchvision 0.19.1 |
| Feature extraction | transformers 4.46.3, open_clip_torch 2.29.0, timm 1.0.30, huggingface-hub 0.36.2 |
| Video and audio I/O | PyAV 18.1.0, OpenCV 4.10, librosa 0.10.2, soundfile 0.12.1, imageio-ffmpeg 0.5.1 |
| Tests | pytest (28 unit tests in `tests/`) |
| Web UI | React 19, Vite 8, Tailwind CSS 4, Node.js 22 |

Hardware: the networks were trained on a CUDA GPU (about 2 minutes per network on an RTX A2000). Inference and all ridge components run on a CPU.

## 8. Running it

| Goal | Command | Time |
|---|---|---|
| Dev score from the shipped component scores | `python combo/c13.py --dev` | seconds |
| Regenerate the submission files | `python combo/c13.py --out out` | seconds |
| Recompute the networks from the raw test EEG | `python scripts/reproduce_c13.py --data /path/to/eeg-mind-reading-challenge` | about a minute (CPU) |
| Same in Docker | `docker run --rm -v "/path/to/eeg-mind-reading-challenge:/data:ro" ghcr.io/sushruths04/neurobytes evaluate` | about a minute |
| Full rebuild of every component | `bash scripts/run_c13_pipeline.sh` | see [PIPELINE.md](PIPELINE.md) |
| Web UI | `docker run --rm -p 127.0.0.1:8080:8080 -v "/path/to/eeg-mind-reading-challenge:/data:ro" ghcr.io/sushruths04/neurobytes-web` | open http://127.0.0.1:8080 |

The web UI shows the predicted and top-ranked videos for every recording, where the true video lands, per-person accuracy, the EEG and its frequency spectrum, and the research behind each design choice. Video captions in the UI were generated with Qwen2.5-VL ([results/captions/README.md](results/captions/README.md)) and are used for display only.

## 9. Repository map

| Path | Contents |
|---|---|
| `combo/` | C13 assembly (`c10.py`, `c11.py`, `c13.py`), networks (`neural_clip.py`, `nc_predict.py`), ours77 and T2 |
| `src/winfuse/` | winfuse: preprocessing, 509 latency-window ridge scorers, cross-participant alignment |
| `src/mindread/` | data loading, encoders/decoders, T2, CSLS |
| `configs/` | frozen C13, T2 and winfuse configurations, model lock file |
| `results/winfuse/` | winfuse fused scores, frozen configuration, measurements |
| `submissions/` | prediction files (C13 and C13 without Sinkhorn) |
| `docker/` | inference image (`ghcr.io/sushruths04/neurobytes`) |
| `neurobyte-web/` | web UI and its image (`ghcr.io/sushruths04/neurobytes-web`) |
| Release `v1.0` | network weights and precomputed video features |
