# NeuroBytes web — EEG → video retrieval frontend

A local web UI for the NeuroDecode ranking. You upload a subject's test recording and
watch the pipeline step through it: the EEG trace of each recording with the model's
latency windows shaded, the video paired with it, and a brain graphic whose 17 nodes
glow with each channel's activity. When it finishes you can browse all 102 recordings
(or switch person), see the top-5 and top-10 videos for each, and export the ranked
list in the submission format.

This copy lives **inside the neurobytes repo** (`neurobytes/neurobyte-web`) and follows
the repo's data-handling convention (see the top-level [README](../README.md) "Data"
section):

- The **repo's own artefacts** are used directly from the repo: `../results/winfuse`
  (winfuse scores, ranked lists, `weights/config.json`) and `../results/neurobytes_models`
  (per-model fused score matrices, generated once by `scripts/export_models.py` from
  `../combo`). These are what the browser ranks with.
- The **official challenge data** (EEG recordings `tst/`, dev labels `tst_stim/`,
  stimulus videos `stimuli_test/`) is **never redistributed in this repository**. Exactly
  like the C13 inference image, it is passed in from outside — natively as
  `DATA=/path/to/eeg-mind-reading-challenge`, and in Docker as a read-only mount at
  `/data`. Without it, the ranking still works (it uses the in-repo scores/matrices); the
  endpoints that serve raw EEG, labels and videos simply return 404.

**Models.** The default is **C13**, the team's final pipeline. The header's model
picker switches to C10 or winfuse (WF); the open recording is re-ranked immediately
and the choice is remembered per browser. (C11/C12 are omitted here because their
network inputs are not shipped in this repo.)

**Pages.**
- `#/` upload, with the selected model's headline numbers.
- `#/analysis` the animated run, then the results.
- `#/analytics` the J1 → C13 progression, the frequency spectrum of the open
  recording, and every experiment, including the rejected ones.
- `#/science` "Why it works". This combines the team's findings with the winfuse
  design choices and their citations (from `results/winfuse/WHY_IT_WORKS.md`).

The sun/moon button switches light/dark mode; light is the default, regardless of the OS setting. The uploaded recording is kept in this
browser's IndexedDB, so a reload keeps you on your results; "Upload another" clears it.

React 19 + Vite 8 + Tailwind CSS 4. The dark theme is Nord, the light theme is tea
green, both from `color-scheme.json`; `theme.test.ts` fails if they disagree.

## Run (from inside the repo)

```bash
cd neurobyte-web
npm ci
# generate the model matrices once, into ../results/neurobytes_models (needs numpy + scikit-learn):
python scripts/export_models.py --team-repo .. --models C13,C10,WF
npm run dev            # http://127.0.0.1:5173
npm test               # vitest
npm run build          # typecheck + production bundle
# production server (no Vite): npm run build && node server/serve.ts   (port 8080)

# to also serve the raw EEG / labels / videos, point DATA at the challenge folder:
DATA=/path/to/eeg-mind-reading-challenge npm run dev
```

## Data

The server prints which paths it is using on startup.

| variable | holds | default |
|---|---|---|
| `DATA` / `NEUROBYTES_DATA_ROOT` | challenge folder with `tst/`, `tst_stim/`, `stimuli_test/` (not in this repo) | `/data` in Docker; a sibling `eeg-mind-reading-challenge` checkout natively |
| `NEUROBYTES_SUBMISSION_DIR` | winfuse: `scores/`, `ranked/`, `weights/config.json` | `../results/winfuse` |
| `NEUROBYTES_MODELS_DIR` | `<model>/<sub>_pre.npy` and `<sub>_ranked.npy` for C13, C10, WF | `../results/neurobytes_models` |
| `NEUROBYTES_LABEL_LOG` | audit log of any ground-truth read | `logs/label_access.log` |

The challenge data is handled exactly as in the top-level README: it is the organizers'
data, so it is never committed and is supplied from outside (`DATA=...` natively, a
`/data:ro` mount in Docker). The expected layout under that folder is `trn/` (full
rebuild only), `tst/` (test recordings `sub-0X_tst.npy`) and `tst_stim/` (dev labels
`sub-0X_tst_stim.npy`); videos live in `stimuli_test/`.

`scripts/export_models.py` writes the models directory from the repo
(`combo/c10.py`, `c13.py` and their shipped inputs). Each `_pre.npy` is the fused
score matrix *before* post-processing; each `_ranked.npy` is the team code's final
ranking, used only to check the browser's result. The script asserts that its C13
output for sub-05/06 is identical to `submissions/C13`. WF (winfuse) also exports; if
the models directory is missing entirely, WF still works by falling back to the
winfuse submission dir.

**Ground truth** (`tst_stim/`) is served for sub-01..04 only, and every read is
appended to `logs/label_access.log`; requests for sub-05/06 labels are always refused.
If the challenge folder is not provided, these endpoints return 404.

`npm test` runs everything that uses synthetic inputs (npy parsing, CSLS, Sinkhorn,
FFT, the HTTP API on a temp tree, theme, content) regardless; the tests that need the
in-repo winfuse scores and exported matrices run once `export_models.py` has been run,
and the ones that need the raw EEG/labels run only when `DATA` points at the challenge
folder.

## What is computed where

| stage | where |
|---|---|
| read `.npy`, validate shape (trials, 17, 1500), SHA-256 fingerprint | browser, on the uploaded file |
| preprocessing, latency windows, ridge scorers, networks, fusion | **offline, in Python** (this repo's `combo/`). The browser loads each model's fused 102 × 102 score matrix |
| CSLS (k = 2, C10/C13) + Sinkhorn (3 iterations, τ = 0.5) + stable argsort | browser: ported from the team's `combo/c10.py` and `mindread.methods.csls` |

`rank.test.ts` checks the browser post-processing against the team code: each shipped
model × 6 subjects, every ranking must match row for row. It also checks that each
model's pooled dev accuracy equals its frozen config (C13 36.03% / 67.16%) when labels
are present. The results page repeats the row check live ("Matches team code 102/102").

Channels are labelled with the electrode names the organisers supplied on 18 Oct
(file order Pz P3 P7 O1 Oz O2 P4 P8 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2, see
`src/lib/montage.ts`). The scalp heatmap places them at the standard 10-10 positions
for those names, interpolates between them, and fades out away from the electrodes.

## Docker

The image holds the built app, the data API (`server/serve.ts`, Node built-ins only,
no Vite at runtime) and the repo's own artefacts (`results/winfuse` and
`results/neurobytes_models`). It holds **no challenge data**: mount the challenge folder
read-only at `/data`, the same way as the C13 inference image. The build context is the
**repo root**, so build from one directory up:

```bash
# from the neurobytes repo root, after running export_models.py once:
docker build -t neurobyte-web -f neurobyte-web/Dockerfile .
D=/path/to/eeg-mind-reading-challenge
docker run --rm -p 127.0.0.1:8080:8080 -v "$D:/data:ro" -v neurobyte-web-logs:/logs neurobyte-web
# open http://127.0.0.1:8080
```

On Windows PowerShell, write the mount as `-v "${D}:/data:ro"`.

- The ranking (scores + model matrices) works even with no `/data` mount, because those
  artefacts are in the repo; the mount only adds the raw EEG, labels and videos.
- `NEUROBYTES_DATA_ROOT` defaults to `/data`; the repo artefacts live at
  `/app/results/...`. Override any of them with the `NEUROBYTES_*` variables above.
- Ground-truth reads are logged to `/logs/label_access.log` (keep the `/logs` volume to retain it).
- Runs as the unprivileged `node` user, with a health check on `/`.
- **Publish on `127.0.0.1` only.** The API has no authentication and serves the EEG,
  videos and dev labels; put it behind an authenticating proxy before exposing it.

## Security

There is no authentication. The dev and preview servers bind to `127.0.0.1` only.
Every file path is built from a fixed pattern (`sub-0[1-6]`, video IDs 1001–1102,
model IDs `C10`–`C13`/`WF`), never from raw input. The tests check path traversal
and HTTP Range edge cases. Don't expose this server to a network as is.
