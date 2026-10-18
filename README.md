# NeuroBytes: EEG-to-video retrieval

**SchubKI National AI Hackathon 2026, EEG mind-reading challenge. Team BierBytes: Sushruth Shivaraju and Ayush Shetty.**

From a 3-second, 17-channel EEG recording, rank the 102 candidate videos so that the video the person was watching comes first.

## Submission

| Official dev test (sub-01..04, 408 recordings) | top-1 | top-5 | mean rank |
|---|---|---|---|
| **NeuroBytes (C13)** | **36.03%** | **67.16%** | 5.99 |
| Chance | 0.98% | 4.90% | 51.5 |

- **Predictions:** `submissions/C13/` (challenge format: one 102 × 102 integer array per test participant, video IDs 1001–1102, rows in the original recording order); fallback without Sinkhorn balancing: `submissions/C13_no_sinkhorn/`.
- **Model weights:** release `v1.0` (`c13o_weights_20261018.tar.gz`, 161 MB).
- **Pretrained models used** (exact names and versions): [MODELS.md](MODELS.md).
- **Technical report** (preprocessing, every component, libraries, analyses): [REPORT.md](REPORT.md).
- **Presentation:** [presentation/BierBytes_Final.pdf](presentation/BierBytes_Final.pdf) ([PowerPoint](presentation/BierBytes_Final.pptx)).

## Data

Only the official challenge data provided by the organizers: EEG recordings, stimulus videos and the development-test labels of sub-01..04. As provided by the organizers:

- **Channels (all subjects):** Pz P3 P7 O1 Oz O2 P4 P8 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2
- **Preprocessing:** band-pass 0.1–40 Hz, blink components removed, downsampled from 1000 to 500 Hz.
- **Recordings:** 3-s epochs (17 × 1500 samples). Test recordings are averaged over repeated viewings.

Expected folder layout (pass its path as `DATA=...`):

```
eeg-mind-reading-challenge/
  trn/        training recordings (needed only for a full rebuild)
  tst/        test recordings, sub-0X_tst.npy
  tst_stim/   development-test labels, sub-0X_tst_stim.npy (needed for the dev score)
```

The videos are not redistributed in this repository; the presentation shows a few frames as screenshots of the web UI.

## Method

| Component | Share | What it does |
|---|---|---|
| winfuse | ~63% | 509 ridge scorers: EEG latency windows (50–600 ms) × embeddings of the provided videos. Cross-participant alignment: each person's model also learns from the other participants' recordings, mapped into that person's channel space. |
| ours77 | ~10% | 77 ridge scorers on video-model features (V-JEPA 2, S3D, CLAP, SigLIP) |
| T2 | ~11% | time-resolved encoding model |
| networks | 15% | two contrastive EEG networks trained on all participants, with a per-person spatial layer and several embedding heads |

The scores are fused by rank, then corrected for hubness (CSLS) and balanced with Sinkhorn over each person's test recordings.

- **Inputs:** every scorer is fitted on the organizers' EEG and on embeddings of the organizers' videos. The frozen pretrained feature extractors are listed in [MODELS.md](MODELS.md).
- **Selection:** settings and fusion weights were chosen by cross-validation on held-out training videos. The development-test labels were used to report results and to compare the few final candidates. Details and limitations: [RESULTS.md](RESULTS.md).

## Quickstart (Windows, macOS, Linux)

You need Python 3.11 and the official challenge folder. The script downloads the network weights (161 MB, checksum verified), recomputes the two networks from your raw test EEG and combines them with the shipped ridge/T2 component scores. It takes about a minute on a CPU.

**macOS / Linux** (Terminal):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/reproduce_c13.py --data /path/to/eeg-mind-reading-challenge
```

**Windows** (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\reproduce_c13.py --data C:\path\to\eeg-mind-reading-challenge
```

Expected output: `DEV-TEST sub-01..04 top1 36.03 top5 67.16 rank 5.99` (rank 6.00 on some CPUs; top-1 and top-5 are identical). The regenerated submission files go to `reproduce/submission/`, and the script checks them against `submissions/C13/`.

- **Recording order:** the test files may be in any order. Each recording is matched to its official row by a fingerprint of its values, so a reordered file gives the same ranking for every recording. A recording that is not in the official files stops the run with a message; rebuild all components with `scripts/run_c13_pipeline.sh` for new data.
- **Weights:** downloaded with `gh` if installed, otherwise over HTTPS. With a local copy, add `--weights /path/to/c13o_weights_20261018` (the extracted folder from release `v1.0`).
- **PowerShell:** if it blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
- **Bash:** on macOS and Linux, `DATA=/path/to/eeg-mind-reading-challenge bash scripts/reproduce_c13.sh` does the same.
- **Verified:** on a clean Linux machine (fresh Python 3.11, `requirements.txt` only), from source and in Docker, and by a teammate on macOS (Apple Silicon): 36.03 / 67.16, with top-1 and top-5 identical to the submission.
- **Library versions:** `requirements.txt` (native) and the pins in `docker/Dockerfile` give the same result. `requirements-winfuse.txt` is needed only to rebuild winfuse from raw data.

## Docker (optional: Windows, macOS, Linux)

A ready-made image (linux/amd64 and linux/arm64) holds the code, weights and pinned libraries. Install Docker Desktop (Windows, macOS) or Docker Engine (Linux), then:

```bash
docker pull ghcr.io/sushruths04/neurobytes:latest
docker run --rm -v "/path/to/eeg-mind-reading-challenge:/data:ro" ghcr.io/sushruths04/neurobytes evaluate
docker run --rm -v "/path/to/eeg-mind-reading-challenge:/data:ro" -v "$PWD/out:/out" ghcr.io/sushruths04/neurobytes predict
```

On Windows PowerShell, write the paths as `-v "C:\path\to\eeg-mind-reading-challenge:/data:ro"` and `-v "${PWD}\out:/out"`. To build the image yourself, see [docker/README.md](docker/README.md).

**Web UI (separate image).** To browse the results (predicted videos, top 10, where the true video lands, per-person accuracy, EEG), run the second image and open http://127.0.0.1:8080:

```bash
docker pull ghcr.io/sushruths04/neurobytes-web:latest
docker run --rm -p 127.0.0.1:8080:8080 -v "/path/to/eeg-mind-reading-challenge:/data:ro" ghcr.io/sushruths04/neurobytes-web
```

The challenge data is mounted, never included in either image. Details: [neurobyte-web/README.md](neurobyte-web/README.md).

## For the organizers

0. **Fastest check: no data, weights or network needed.** It runs in seconds from the shipped component scores:
   ```bash
   python combo/c13.py --dev          # 36.03 / 67.16 / 5.99
   python combo/c13.py --out out      # regenerates the submission files, byte-identical to submissions/C13/
   ```
1. **Submitted predictions:** `submissions/C13/`. To check the format: `python scripts/check_submission.py submissions/C13`.
2. **Regenerate them from the raw test EEG:** use the quickstart or Docker (above).
3. **Retrain everything**, optionally with more training data: `scripts/run_c13_pipeline.sh` (2.8 GB download, about 6 GB of free disk; a CUDA GPU is recommended, see [PIPELINE.md](PIPELINE.md)).
4. **Pretrained models used:** [MODELS.md](MODELS.md).

## Repository map

| Path | What |
|---|---|
| `combo/` | C13 assembly (`c10.py`, `c11.py`, `c13.py`), networks (`neural_clip.py`, `nc_predict.py`), ours77 and T2 producers |
| `src/winfuse/` | winfuse: 509 latency-window ridge scorers + cross-subject alignment (`compare_embed/cross_subject.py`) |
| `src/mindread/` | mindread: data loading, encoders/decoders, T2, CSLS |
| `results/winfuse/` | aligned winfuse fused scores, frozen config, evidence |
| `results/captions/` | Qwen2.5-VL captions of the test videos, shown in the web UI only ([how they were made](results/captions/README.md)) |
| `combo/c10_inputs/`, `c13_inputs/` | shipped component scores used by the quickstart |
| `submissions/` | submitted prediction files (C13 and the fallback without Sinkhorn) |
| `scripts/` | `reproduce_c13.sh` (quickstart), `run_c13_pipeline.sh` (full rebuild), feature extraction, format check |
| `docker/` | CPU inference image (`build.sh`, `Dockerfile`, `entrypoint.sh`) |
| `neurobyte-web/` | web UI for the results, with its own image `ghcr.io/sushruths04/neurobytes-web` (see its README) |
| `configs/` | frozen C13 configuration, T2 settings, pretrained-model lock file |
| `verification/` | checksums and the clean-machine verification record |
| `tests/` | unit tests (`pytest tests`) |
| [RESULTS.md](RESULTS.md) · [PIPELINE.md](PIPELINE.md) · [MODELS.md](MODELS.md) | results and robustness · full rebuild · pretrained models |


## License

NeuroBytes Source-Available License; see [LICENSE](LICENSE). You may view the code. The challenge organizers and jury may run and reproduce it for evaluation. Any other use, copying or redistribution requires the owners' written permission.
