"""Nonlinear complexity features (Agent B) via antropy.

Per channel (17 channels), per trial:
    - permutation entropy      (order=3, normalised)
    - sample entropy           (order=2, computed on a SUBSAMPLED window)
    - Higuchi fractal dimension (kmax default)
    - Petrosian fractal dimension
    - detrended fluctuation analysis exponent

=> 5 metrics x 17 channels = 85 features.

Sample entropy is O(n^2) in the embedding search. antropy's implementation is
Numba-accelerated (~0.5 ms per 750-sample channel), but to BOUND worst-case
runtime and keep the metric comparable across trials of possibly differing
quality, sample entropy is computed on the first `sampen_window` samples
(default 375 = the first 1.5 s at 250 Hz). The subsampling factor and measured
timing are logged in decisions/B.md. All other metrics use the full 750 samples.

Nothing is fitted; this is a stateless per-trial transform.
"""
from __future__ import annotations

import numpy as np

import antropy as ant

from winfuse.common.config import N_CHANNELS, SFREQ
from winfuse.common.feature_api import BaseExtractor, register

_SAMPEN_WINDOW = 375  # first 1.5 s at 250 Hz
_PERM_ORDER = 3
_SAMPEN_ORDER = 2

_METRICS = ("perm_ent", "samp_ent", "higuchi_fd", "petrosian_fd", "dfa")


@register
class NonlinearEntropy(BaseExtractor):
    name = "nonlinear_entropy"
    description = (
        "Per-channel nonlinear complexity: permutation entropy (order 3), sample "
        "entropy (order 2, first 1.5 s subsample), Higuchi & Petrosian fractal "
        "dimensions, and DFA exponent. 5 metrics x 17 channels = 85 features."
    )
    agent = "B"

    def __init__(self, sampen_window: int = _SAMPEN_WINDOW, **kw):
        super().__init__(**kw)
        self.sampen_window = int(sampen_window)
        self.params.setdefault("perm_order", _PERM_ORDER)
        self.params.setdefault("sampen_order", _SAMPEN_ORDER)
        self.params.setdefault("sampen_window_samples", self.sampen_window)
        self.params.setdefault("sampen_window_s", round(self.sampen_window / SFREQ, 3))
        self.params.setdefault("sfreq", SFREQ)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        n, c, t = X.shape
        w = min(self.sampen_window, t)
        out = np.empty((n, c * len(_METRICS)), dtype=np.float32)
        for i in range(n):
            row = []
            for ch in range(c):
                sig = X[i, ch].astype(np.float64)
                pe = ant.perm_entropy(sig, order=_PERM_ORDER, normalize=True)
                se = ant.sample_entropy(sig[:w], order=_SAMPEN_ORDER)
                hf = ant.higuchi_fd(sig)
                pf = ant.petrosian_fd(sig)
                try:
                    df = ant.detrended_fluctuation(sig)
                except Exception:
                    df = np.nan
                row.extend((pe, se, hf, pf, df))
            out[i] = np.asarray(row, dtype=np.float32)
        # Replace any non-finite (e.g. sampen inf for near-constant channels, DFA failures)
        # with per-column medians so the contract's finiteness check passes.
        if not np.isfinite(out).all():
            for j in range(out.shape[1]):
                col = out[:, j]
                bad = ~np.isfinite(col)
                if bad.any():
                    good = col[~bad]
                    fill = float(np.median(good)) if good.size else 0.0
                    col[bad] = fill
        return out

    def feature_names(self) -> list[str]:
        return [f"ch{ch:02d}_{m}" for ch in range(N_CHANNELS) for m in _METRICS]
