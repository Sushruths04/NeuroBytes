"""EEG target construction. All operations are per recording (no statistics fitted across data)."""
from __future__ import annotations

import numpy as np
from scipy import signal

FS = 500


def to_100hz(X: np.ndarray) -> np.ndarray:
    """(n, 17, 1500) at 500 Hz -> (n, 17, 300) at 100 Hz, anti-aliased (same as the organizer notebook)."""
    return signal.decimate(np.asarray(X, dtype=np.float64), 5, axis=-1, ftype='fir', zero_phase=True)


def remove_offset(X: np.ndarray, ms: float = 50.0) -> np.ndarray:
    """Subtract each channel's mean over the first `ms` (pre-visual period; no baseline exists)."""
    k = int(round(ms / 1000 * FS))
    return X - X[..., :k].mean(-1, keepdims=True)


def target(X: np.ndarray, t0_ms: float = 50, t1_ms: float = 1000, offset: bool = True,
           fs_out: int = 100, channels=None, smooth_hz: float | None = None) -> np.ndarray:
    """Flattened channel x time target over [t0, t1) at fs_out (100, 50 or 25 Hz).

    channels: optional list of channel indices; smooth_hz: optional zero-phase low-pass before resampling.
    """
    if offset:
        X = remove_offset(X)
    x = to_100hz(X)
    if smooth_hz:
        x = signal.sosfiltfilt(signal.butter(4, smooth_hz, fs=100, output='sos'), x, axis=-1)
    if channels is not None:
        x = x[:, list(channels)]
    a, b = int(t0_ms / 10), int(t1_ms / 10)
    x = x[..., a:b]
    if fs_out != 100:
        k = 100 // fs_out
        n = x.shape[-1] // k
        x = x[..., :n * k].reshape(*x.shape[:-1], n, k).mean(-1)   # bin-average (anti-aliased by the low-pass)
    return x.reshape(len(X), -1)


def slow_target(X: np.ndarray, t0_ms: float = 500, t1_ms: float = 3000, cutoff_hz: float = 4.0, fs_out: int = 10) -> np.ndarray:
    """Slow (<cutoff) time-locked EEG over [t0, t1) at fs_out Hz, flattened (ch x t). Offset removed as in `target`."""
    x = to_100hz(remove_offset(X))
    x = signal.sosfiltfilt(signal.butter(4, cutoff_hz, fs=100, output='sos'), x, axis=-1)
    x = x[..., int(t0_ms / 10):int(t1_ms / 10):100 // fs_out]
    return x.reshape(len(X), -1)


def two_timescale_target(X, onset_kw=None, slow_kw=None):
    """[onset target | slow target]; the split index is 17 * n_onset_samples."""
    onset_kw = onset_kw or dict(t0_ms=50, t1_ms=1000)
    return np.concatenate([target(X, **onset_kw), slow_target(X, **(slow_kw or {}))], 1)
