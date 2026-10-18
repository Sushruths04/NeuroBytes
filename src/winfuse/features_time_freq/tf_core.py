"""Time-resolved band power core (Agent A).

Computes band-limited power in ~250 ms sliding windows (125 ms step) using
Morlet-wavelet convolution per band. CPU-only, float32-friendly.

Design (see decisions A.md):
* One complex Morlet per band, centred at the band's geometric mean frequency,
  with n_cycles chosen so the wavelet's time support is ~250 ms (TF_WINDOW_S).
* Instantaneous power = |conv|^2, then averaged inside each sliding window.
* Windows: length TF_WINDOW_S (250 ms = 62 samples), step TF_STEP_S (125 ms = 31
  samples) across the 3 s / 750-sample epoch -> 23 windows.
"""
from __future__ import annotations

import numpy as np

from winfuse.common.config import BANDS, SFREQ, TF_STEP_S, TF_WINDOW_S

# Morlet cycles per band. Lower-frequency bands get fewer cycles so the wavelet
# stays near ~250 ms; higher bands get more cycles for better frequency focus.
# n_cycles = f_center * TF_WINDOW_S, floored at 3 for numerical stability.
WIN_SAMPLES = int(round(TF_WINDOW_S * SFREQ))   # 62
STEP_SAMPLES = int(round(TF_STEP_S * SFREQ))    # 31


def band_centers() -> dict[str, float]:
    """Geometric-mean centre frequency per band."""
    return {b: float(np.sqrt(lo * hi)) for b, (lo, hi) in BANDS.items()}


def n_cycles_for(fc: float) -> float:
    return max(3.0, fc * TF_WINDOW_S)


def _morlet(fc: float, n_cycles: float, sfreq: float = SFREQ) -> np.ndarray:
    """Complex Morlet wavelet, L2-normalised, centred at fc."""
    sigma_t = n_cycles / (2.0 * np.pi * fc)
    # +/- 3 sigma support
    half = int(round(3.0 * sigma_t * sfreq))
    t = np.arange(-half, half + 1) / sfreq
    gauss = np.exp(-(t ** 2) / (2.0 * sigma_t ** 2))
    carrier = np.exp(2j * np.pi * fc * t)
    w = gauss * carrier
    w = w - w.mean()  # remove DC leakage
    w = w / np.sqrt((np.abs(w) ** 2).sum())
    return w.astype(np.complex64)


def _window_edges(n_samples: int) -> list[tuple[int, int]]:
    edges = []
    start = 0
    while start + WIN_SAMPLES <= n_samples:
        edges.append((start, start + WIN_SAMPLES))
        start += STEP_SAMPLES
    return edges


def tf_band_power(X: np.ndarray) -> tuple[np.ndarray, list[str], list[tuple[int, int]]]:
    """Time-resolved band power.

    X : (n_trials, 17, n_samples) float32
    Returns
    -------
    power : (n_trials, 17, n_bands, n_windows) float32  (log10 power)
    bands : list of band names (order matches axis 2)
    edges : list of (start, stop) sample windows (axis 3)
    """
    from scipy.signal import fftconvolve

    n_trials, n_ch, n_s = X.shape
    centers = band_centers()
    bands = list(BANDS)
    edges = _window_edges(n_s)
    n_bands = len(bands)
    n_win = len(edges)

    out = np.empty((n_trials, n_ch, n_bands, n_win), dtype=np.float32)
    Xr = X.reshape(n_trials * n_ch, n_s).astype(np.float32)

    for bi, b in enumerate(bands):
        w = _morlet(centers[b], n_cycles_for(centers[b]))
        # convolve each row with the wavelet (same length), take |.|^2
        conv = fftconvolve(Xr, w[None, :], mode="same", axes=-1)
        inst_pow = (conv.real ** 2 + conv.imag ** 2).astype(np.float32)
        for wi, (s0, s1) in enumerate(edges):
            out[:, :, bi, wi] = inst_pow[:, s0:s1].mean(axis=-1).reshape(n_trials, n_ch)

    out = np.log10(out + 1e-12).astype(np.float32)
    names = [b for b in bands]
    return out, names, edges
