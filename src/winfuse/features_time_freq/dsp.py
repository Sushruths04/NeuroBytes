"""Shared DSP helpers for Agent A's time/frequency extractors.

Everything here operates on the preprocessed input convention: float32 epochs of
shape ``(n_trials, 17, 750)`` at ``SFREQ`` = 250 Hz. All functions are stateless.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from winfuse.common.config import BANDS, SFREQ

# Welch segment length. See decision "Welch nperseg / segment length".
# 1.0 s Hann segment = 250 samples at 250 Hz -> 1 Hz bins, with 50% overlap
# giving ~5 overlapping segments across the 3 s epoch.
WELCH_NPERSEG = 250
WELCH_NOVERLAP = 125


def welch_psd(X: np.ndarray, nperseg: int = WELCH_NPERSEG,
              noverlap: int = WELCH_NOVERLAP) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD over the last axis.

    Returns (freqs (F,), psd (n_trials, 17, F)) float32/float64.
    """
    freqs, psd = welch(
        X, fs=SFREQ, window="hann", nperseg=nperseg, noverlap=noverlap,
        detrend="constant", axis=-1, scaling="density",
    )
    return freqs.astype(np.float64), psd.astype(np.float64)


def band_mask(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Boolean mask selecting [lo, hi) frequency bins (hi inclusive at the top edge)."""
    return (freqs >= lo) & (freqs <= hi)


def integrate_band(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Trapezoidal integral of psd over [lo, hi]. psd shape (..., F) -> (...,)."""
    m = band_mask(freqs, lo, hi)
    if m.sum() < 2:
        # too few bins to integrate; fall back to the mean density * bandwidth
        return psd[..., m].mean(axis=-1) * (hi - lo)
    return np.trapz(psd[..., m], freqs[m], axis=-1)


def all_band_powers(freqs: np.ndarray, psd: np.ndarray) -> dict[str, np.ndarray]:
    """Absolute power per band. Returns {band: (n_trials, 17)}."""
    return {b: integrate_band(freqs, psd, lo, hi) for b, (lo, hi) in BANDS.items()}
