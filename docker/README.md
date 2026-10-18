# NeuroBytes C13 in Docker

The image combines the h6/h10 network weights, code, pinned CPU libraries and shipped ridge/T2 component scores. Mount the official challenge data to recompute network scores and assemble predictions. The weight release is `v1.0`, archive `c13o_weights_20261018.tar.gz`.

Docker is optional: the same results come from the source quickstart in the main [README](../README.md). The image works on Windows (Docker Desktop), on Intel and Apple Silicon Macs, and on Linux.

## 1. Get the image

**Pull the ready-made image** (recommended; built for linux/amd64 and linux/arm64):

```bash
docker pull ghcr.io/sushruths04/neurobytes:latest
```

**Or build it yourself** from a git clone or a "Download ZIP" copy:

```bash
bash docker/build.sh                                                      # downloads the weights with gh
WEIGHTS_TAR=/path/to/c13o_weights_20261018.tar.gz bash docker/build.sh   # or use a local copy of the weights
```

On Windows, run `build.sh` in WSL or Git Bash. A git clone packages the committed checkout; uncommitted changes are left out, with a warning.

## 2. Run

macOS / Linux:

```bash
D=/path/to/eeg-mind-reading-challenge
docker run --rm -v "$D:/data:ro" ghcr.io/sushruths04/neurobytes evaluate
docker run --rm -v "$D:/data:ro" -v "$PWD/out:/out" ghcr.io/sushruths04/neurobytes predict
docker run --rm -v "$PWD/out:/out" ghcr.io/sushruths04/neurobytes replay
docker run --rm -p 127.0.0.1:8080:8080 -v "$D:/data:ro" -v "$PWD/out:/out" ghcr.io/sushruths04/neurobytes web
```

Windows (PowerShell):

```powershell
$D = "C:\path\to\eeg-mind-reading-challenge"
docker run --rm -v "${D}:/data:ro" ghcr.io/sushruths04/neurobytes evaluate
docker run --rm -v "${D}:/data:ro" -v "${PWD}\out:/out" ghcr.io/sushruths04/neurobytes predict
```

| Command | Behavior |
|---|---|
| `predict` | Recompute h6/h10 scores from mounted test EEG, combine with shipped component scores, write the ranked candidate lists to `/out`. |
| `evaluate` | Score development recordings sub-01..04; expected 36.03% top-1, 67.16% top-5, mean rank 6.00 on CPU. |
| `replay` | Copy the frozen prediction lists; no model inference. |
| `web` | Serve the neurobyte-web UI + read-only data API on port 8080 (override with `PORT`). Ranking uses the baked-in winfuse scores and model matrices; the `/data:ro` mount adds the raw EEG, dev labels and stimulus videos. Publish on `127.0.0.1` only (no auth). See [../neurobyte-web/README.md](../neurobyte-web/README.md). |

Predictions are 102 × 102 integer arrays. Each row ranks video IDs 1001–1102 in the original recording order. CSLS and Sinkhorn operate jointly over a person's test recordings; the no-Sinkhorn fallback still uses batch hub correction.

## Scope and verification

Only the neural branch is recomputed from mounted EEG in `predict`/`evaluate`. The aligned winfuse, ours77 and T2 scores are shipped caches. For training and full-component rebuild instructions and known discrepancies, see [PIPELINE.md](../PIPELINE.md).

Verified on 18 Oct 2026: the image built from this Dockerfile on a clean Modal machine gives `evaluate` 36.03% top-1, 67.16% top-5, mean rank 6.00, and `predict` matches `submissions/C13/` in top-1 and top-5 for every recording.
