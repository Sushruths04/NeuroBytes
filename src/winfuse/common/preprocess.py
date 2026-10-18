"""Preprocessing. FROZEN CONTRACT - sub-agents must not edit this file.

Pipeline, in order
------------------
1. 50 Hz notch            (IIR notch, Q=30, zero-phase filtfilt)
2. 0.5-40 Hz band-pass    (Butterworth order 4, zero-phase sosfiltfilt, odd padding)
3. decimate 500 -> 250 Hz (scipy.signal.decimate, FIR anti-alias, zero_phase=True)
4. baseline handling      (default: none -- see the decision note below)
5. per-channel z-score    (mean/std fitted on TRAINING trials only)

Steps 1-4 are stateless: they contain no learned parameter, so applying them to
test data cannot leak anything. Step 5 is the only fitted step, and `fit` only
ever sees training epochs.

Why `baseline="none"` is the default
------------------------------------
The recording starts at video onset, so there is no pre-stimulus period to use as
a baseline. The options considered, and why the default is what it is, are logged
in decisions/orchestrator.md ("Baseline and drift handling without a pre-stimulus
period"). Short version: the 0.5 Hz zero-phase high-pass already removes slow
drift, and every alternative baseline either injects noise (a 40 ms pseudo-baseline
is only 10 samples at 250 Hz) or removes signal (whole-epoch mean subtraction
deletes the sustained response, which over a 3 s video is real signal). The
alternatives are implemented and selectable so that any agent can test the
sensitivity of its result to this choice.

Usage
-----
    from winfuse.common import preprocess as pp

    # cached, fitted on train only -- this is what you normally want
    Xtr, ids = pp.preprocessed_train("sub-01")        # (n_trials, 17, 750) float32
    prep     = pp.get_preprocessor("sub-01")          # the fitted object

    # test data, dev subjects only, access is logged
    Xte = pp.preprocessed_test("sub-01", reason="final dev validation")
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, decimate, iirnotch, filtfilt, sosfiltfilt

from . import io as _io
from .config import (
    BANDPASS,
    CACHE_DIR,
    DECIM,
    FILTER_ORDER,
    NOTCH_FREQ,
    NOTCH_Q,
    N_SAMPLES,
    N_SAMPLES_RAW,
    SFREQ,
    SFREQ_RAW,
)

BASELINE_MODES = ("none", "onset_proxy", "detrend", "epoch_mean")
_PREPROC_CACHE = CACHE_DIR / "preproc"
_PREPROC_CACHE.mkdir(parents=True, exist_ok=True)

# Earliest plausible cortical visual response. Used by baseline="onset_proxy".
ONSET_PROXY_END_S = 0.040


@dataclass
class PreprocConfig:
    sfreq_raw: float = SFREQ_RAW
    bandpass: tuple[float, float] = BANDPASS
    notch_freq: float | None = NOTCH_FREQ
    notch_q: float = NOTCH_Q
    order: int = FILTER_ORDER
    decim: int = DECIM
    baseline: str = "none"
    zscore: bool = True

    def __post_init__(self) -> None:
        if self.baseline not in BASELINE_MODES:
            raise ValueError(f"baseline must be one of {BASELINE_MODES}, got {self.baseline!r}")
        lo, hi = self.bandpass
        nyq_out = self.sfreq_raw / self.decim / 2
        if hi >= nyq_out:
            raise ValueError(
                f"band-pass high edge {hi} Hz must be below the post-decimation "
                f"Nyquist {nyq_out} Hz to avoid aliasing"
            )

    @property
    def sfreq(self) -> float:
        return self.sfreq_raw / self.decim

    def key(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]


# --------------------------------------------------------------------- stateless filtering
def _notch(X: np.ndarray, cfg: PreprocConfig) -> np.ndarray:
    if cfg.notch_freq is None:
        return X
    if cfg.notch_freq >= cfg.sfreq_raw / 2:
        return X
    b, a = iirnotch(cfg.notch_freq, cfg.notch_q, fs=cfg.sfreq_raw)
    return filtfilt(b, a, X, axis=-1, padtype="odd").astype(np.float32, copy=False)


def _bandpass(X: np.ndarray, cfg: PreprocConfig) -> np.ndarray:
    lo, hi = cfg.bandpass
    sos = butter(cfg.order, [lo, hi], btype="bandpass", fs=cfg.sfreq_raw, output="sos")
    # padtype='odd' reflects-and-negates at the edges, which is the mildest option for
    # a 3 s epoch high-passed at 0.5 Hz (only 1.5 cycles of the cut-off period fit in
    # the epoch, so edge transients are unavoidable -- see decisions/orchestrator.md).
    return sosfiltfilt(sos, X, axis=-1, padtype="odd").astype(np.float32, copy=False)


def _decimate(X: np.ndarray, cfg: PreprocConfig) -> np.ndarray:
    if cfg.decim == 1:
        return X
    return decimate(X, cfg.decim, axis=-1, ftype="fir", zero_phase=True).astype(
        np.float32, copy=False
    )


def _baseline(X: np.ndarray, cfg: PreprocConfig) -> np.ndarray:
    mode = cfg.baseline
    if mode == "none":
        return X
    if mode == "epoch_mean":
        return (X - X.mean(axis=-1, keepdims=True)).astype(np.float32, copy=False)
    if mode == "onset_proxy":
        n = max(int(round(ONSET_PROXY_END_S * cfg.sfreq)), 2)
        return (X - X[..., :n].mean(axis=-1, keepdims=True)).astype(np.float32, copy=False)
    if mode == "detrend":
        n = X.shape[-1]
        t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        tt = float(t @ t)
        slope = (X * t).sum(axis=-1, keepdims=True) / tt
        inter = X.mean(axis=-1, keepdims=True)
        return (X - (inter + slope * t)).astype(np.float32, copy=False)
    raise ValueError(mode)


def filter_epochs(X: np.ndarray, cfg: PreprocConfig | None = None) -> np.ndarray:
    """Stateless part of the pipeline: notch -> band-pass -> decimate -> baseline.

    Contains no fitted parameter, so this is safe to apply to any data.
    """
    cfg = cfg or PreprocConfig()
    X = np.asarray(X, dtype=np.float32)
    _io.assert_epochs(X, n_samples=N_SAMPLES_RAW if cfg.sfreq_raw == SFREQ_RAW else None)
    X = _notch(X, cfg)
    X = _bandpass(X, cfg)
    X = _decimate(X, cfg)
    X = _baseline(X, cfg)
    return np.ascontiguousarray(X, dtype=np.float32)


# --------------------------------------------------------------------- the fitted object
class Preprocessor:
    """Stateless filtering + per-channel z-scoring whose stats come from train only."""

    def __init__(self, cfg: PreprocConfig | None = None) -> None:
        self.cfg = cfg or PreprocConfig()
        self.mean_: np.ndarray | None = None  # (17, 1)
        self.std_: np.ndarray | None = None   # (17, 1)
        self.n_fit_trials_: int | None = None
        self.fitted_on_: str | None = None

    # -- introspection
    @property
    def sfreq(self) -> float:
        return self.cfg.sfreq

    @property
    def n_samples(self) -> int:
        return N_SAMPLES_RAW // self.cfg.decim

    def times(self) -> np.ndarray:
        return _io.times(self.n_samples, self.sfreq)

    def is_fitted(self) -> bool:
        return self.mean_ is not None

    # -- api
    def fit(self, X_train: np.ndarray, subject: str | None = None) -> "Preprocessor":
        """Fit z-score statistics. X_train must be RAW training epochs (n, 17, 1500)."""
        Xf = filter_epochs(X_train, self.cfg)
        # Pool over trials and time; one mean/std per channel.
        self.mean_ = Xf.mean(axis=(0, 2), keepdims=False).astype(np.float32).reshape(-1, 1)
        std = Xf.std(axis=(0, 2), keepdims=False).astype(np.float32).reshape(-1, 1)
        self.std_ = np.maximum(std, 1e-8).astype(np.float32)
        self.n_fit_trials_ = int(X_train.shape[0])
        self.fitted_on_ = subject
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the full pipeline. Returns float32 (n_trials, 17, n_samples)."""
        if self.cfg.zscore and not self.is_fitted():
            raise RuntimeError("Preprocessor.transform called before fit()")
        Xf = filter_epochs(X, self.cfg)
        if self.cfg.zscore:
            Xf = (Xf - self.mean_[None, :, :]) / self.std_[None, :, :]
        return np.ascontiguousarray(Xf, dtype=np.float32)

    def fit_transform(self, X_train: np.ndarray, subject: str | None = None) -> np.ndarray:
        return self.fit(X_train, subject=subject).transform(X_train)

    def describe(self) -> dict:
        return {
            "config": asdict(self.cfg),
            "sfreq_out": self.sfreq,
            "n_samples_out": self.n_samples,
            "fitted": self.is_fitted(),
            "fitted_on": self.fitted_on_,
            "n_fit_trials": self.n_fit_trials_,
            "channel_mean": None if self.mean_ is None else self.mean_.ravel().round(6).tolist(),
            "channel_std": None if self.std_ is None else self.std_.ravel().round(6).tolist(),
        }


# --------------------------------------------------------------------- cached convenience
def _cache_path(sub: str, split: str, cfg: PreprocConfig) -> Path:
    return _PREPROC_CACHE / f"{sub}_{split}_{cfg.key()}.npy"


def get_preprocessor(sub: str | int, cfg: PreprocConfig | None = None) -> Preprocessor:
    """Fit (or load from cache) the Preprocessor for a subject, using TRAIN data only."""
    import joblib

    cfg = cfg or PreprocConfig()
    sub = _io._norm_sub(sub)
    p = _PREPROC_CACHE / f"{sub}_prep_{cfg.key()}.joblib"
    if p.exists():
        prep = joblib.load(p)
        if isinstance(prep, Preprocessor) and prep.is_fitted():
            return prep
    Xtr, _ = _io.load_subject_train(sub)
    prep = Preprocessor(cfg).fit(Xtr, subject=sub)
    joblib.dump(prep, p, compress=3)
    del Xtr
    return prep


def preprocessed_train(
    sub: str | int, cfg: PreprocConfig | None = None, use_cache: bool = True
) -> tuple[np.ndarray, list[int]]:
    """Preprocessed training epochs + stimulus IDs. Cached to cache/preproc/."""
    cfg = cfg or PreprocConfig()
    sub = _io._norm_sub(sub)
    cp = _cache_path(sub, "train", cfg)
    ip = cp.with_suffix(".ids.json")
    if use_cache and cp.exists() and ip.exists():
        X = np.load(cp, mmap_mode=None)
        ids = json.loads(ip.read_text())
        return np.ascontiguousarray(X, dtype=np.float32), ids
    Xraw, ids = _io.load_subject_train(sub)
    prep = get_preprocessor(sub, cfg)
    X = prep.transform(Xraw)
    del Xraw
    if use_cache:
        np.save(cp, X)
        ip.write_text(json.dumps(ids))
    return X, ids


def preprocessed_test(
    sub: str | int,
    reason: str,
    cfg: PreprocConfig | None = None,
    use_cache: bool = True,
    inference: bool = False,
) -> np.ndarray:
    """Preprocessed TEST epochs, using the TRAIN-fitted scaler.

    `reason` is mandatory and is written to logs/test_access.log.

    With `inference=False` (the default) sub-05/06 are refused by src.common.io, which
    is what keeps every fitting path away from them.

    With `inference=True` the held-out subjects' RECORDINGS may be read, because the
    challenge requires ranking the candidates for each of them. The scaler is still
    fitted on that subject's own TRAINING recordings only, and their labels remain
    unreachable. See src.common.io.load_subject_test_for_inference for the rules this
    follows.
    """
    cfg = cfg or PreprocConfig()
    sub = _io._norm_sub(sub)
    cp = _cache_path(sub, "test", cfg)
    if use_cache and cp.exists():
        return np.ascontiguousarray(np.load(cp), dtype=np.float32)
    if inference:
        Xraw = _io.load_subject_test_for_inference(sub, reason=reason)
    else:
        Xraw, _ = _io.load_subject_test(sub, reason=reason)
    prep = get_preprocessor(sub, cfg)   # fitted on train only
    X = prep.transform(Xraw)
    del Xraw
    if use_cache:
        np.save(cp, X)
    return X


def window_slice(t0: float, t1: float, sfreq: float | None = None) -> slice:
    """Sample slice for a [t0, t1) window in seconds, t=0 at onset."""
    sfreq = sfreq or SFREQ
    i0 = max(int(np.floor(t0 * sfreq)), 0)
    i1 = min(int(np.ceil(t1 * sfreq)), N_SAMPLES)
    if i1 <= i0:
        i1 = i0 + 1
    return slice(i0, i1)
