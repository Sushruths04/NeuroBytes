"""Shared constants. FROZEN CONTRACT - sub-agents must not edit this file.

Everything that more than one agent needs to agree on lives here: paths, sampling
rates, frequency bands, ERP windows, the random seed and the worker cap.

If you are a sub-agent and you need a change here, request it via the orchestrator.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
# NOTE: the brief says the data is in ./data. It is not; it sits at the repo root.
# See reports/00_data_inspection.md section 0.
def _resolve_root() -> Path:
    """Locate the challenge data.

    Order of precedence:
      1. $WINFUSE_DATA_ROOT                      - explicit override
      2. `challenge_root` in configs/paths.yaml  - this repo's convention
      3. the repository root                     - original layout, data at root

    The chosen directory must contain trn/ and tst/. This is the only place that
    knows where the data lives; nothing else in winfuse hard-codes a path.
    """
    import os

    env = os.environ.get("WINFUSE_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    repo = here.parents[3]            # <repo>/src/winfuse/common/config.py
    cfg = repo / "configs" / "paths.yaml"
    if cfg.exists():
        try:
            import yaml

            raw = yaml.safe_load(cfg.read_text()) or {}
            cr = raw.get("challenge_root")
            if cr:
                cand = (repo / cr).resolve() if not Path(cr).is_absolute() else Path(cr)
                if (cand / "trn").is_dir():
                    return cand
        except Exception:
            pass
    return repo


ROOT = _resolve_root()
WRITE_ROOT = Path(__file__).resolve().parents[3]

TRN_DIR = ROOT / "trn"
TST_DIR = ROOT / "tst"
TST_STIM_DIR = ROOT / "tst_stim"
STIM_TRAIN_DIR = ROOT / "stimuli_train"
STIM_TEST_DIR = ROOT / "stimuli_test"

CACHE_DIR = WRITE_ROOT / "cache"
RESULTS_DIR = WRITE_ROOT / "results"
FIGURES_DIR = WRITE_ROOT / "figures"
REPORTS_DIR = WRITE_ROOT / "reports"
DECISIONS_DIR = WRITE_ROOT / "decisions"
LOGS_DIR = WRITE_ROOT / "logs"

for _d in (CACHE_DIR, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR, DECISIONS_DIR, LOGS_DIR):
    _d.mkdir(exist_ok=True)

# ---------------------------------------------------------------- subjects
DEV_SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-04")
HELDOUT_SUBJECTS = ("sub-05", "sub-06")
ALL_SUBJECTS = DEV_SUBJECTS + HELDOUT_SUBJECTS

# ---------------------------------------------------------------- signal
SFREQ_RAW = 500.0          # Hz. Documented in the challenge + baseline notebook, NOT in the files.
N_CHANNELS = 17
N_SAMPLES_RAW = 1500       # 3.0 s at 500 Hz

DECIM = 2                  # 500 -> 250 Hz
SFREQ = SFREQ_RAW / DECIM  # 250.0 Hz, the rate every extractor sees
N_SAMPLES = N_SAMPLES_RAW // DECIM  # 750

EPOCH_TMIN = 0.0           # t=0 is video onset. There is NO pre-stimulus period.
EPOCH_TMAX = N_SAMPLES_RAW / SFREQ_RAW  # 3.0 s

# Band-pass / notch
BANDPASS = (0.5, 40.0)     # Hz
NOTCH_FREQ = 50.0          # Hz (mains). Already outside the pass-band; applied for explicitness.
NOTCH_Q = 30.0
FILTER_ORDER = 4           # Butterworth order per direction (sosfiltfilt doubles it)

# Candidate stimulus pools
TRAIN_STIM_RANGE = (1, 1000)
TEST_STIM_RANGE = (1001, 1102)
N_TEST_TRIALS = 102

# ---------------------------------------------------------------- frequency bands
# Upper edge of gamma is capped at 40 Hz by the band-pass above.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

# ---------------------------------------------------------------- ERP windows (seconds from onset)
# Named after the classic components they are *expected* to contain. Because the
# channels are anonymous and there is no baseline, these are windows of interest,
# not verified component measurements.
ERP_WINDOWS: dict[str, tuple[float, float]] = {
    "p1n1": (0.080, 0.150),
    "n2p2": (0.150, 0.300),
    "late": (0.300, 0.600),
    "w600_1000": (0.600, 1.000),
    "w1000_1500": (1.000, 1.500),
    "w1500_2000": (1.500, 2.000),
    "w2000_2500": (2.000, 2.500),
    "w2500_3000": (2.500, 3.000),
}

# Window used by the baseline notebook for retrieval (samples 5:60 at 100 Hz = 0.05-0.60 s)
RETRIEVAL_WINDOW = (0.05, 0.60)

# Sliding window for time-resolved band power
TF_WINDOW_S = 0.250
TF_STEP_S = 0.125

# ---------------------------------------------------------------- compute budget
SEED = 1234
N_JOBS = 4                 # hard cap per the brief
FLOAT_DTYPE = "float32"

# ---------------------------------------------------------------- stimulus descriptors
DESCRIPTOR_NAMES = (
    "audio_rms_mean",
    "audio_rms_std",
    "audio_onset_strength",
    "luminance_mean",
    "luminance_std",
    "motion_energy_mean",
    "motion_energy_std",
    "contrast_mean",
)
