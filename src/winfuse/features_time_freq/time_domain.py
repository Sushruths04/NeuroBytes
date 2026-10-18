"""Time-domain extractors (Agent A).

* raw_signal   — decimated preprocessed trace, flattened
* erp_windows  — mean amplitude per channel per ERP window
* hjorth       — activity / mobility / complexity per channel
* time_stats   — ptp, skew, kurtosis, line length, RMS, zero-crossing rate per channel
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from winfuse.common.config import ERP_WINDOWS, N_CHANNELS, SFREQ
from winfuse.common.feature_api import BaseExtractor, channel_feature_names, register
from winfuse.common.preprocess import window_slice

# ------------------------------------------------------------------ raw_signal
# Decimation factor applied on top of the 250 Hz preprocessed signal.
# 250 Hz / 5 = 50 Hz -> 750 / 5 = 150 samples/channel. See decision note.
RAW_DECIM = 5


@register
class RawSignal(BaseExtractor):
    name = "raw_signal"
    description = "Preprocessed trace block-averaged to ~50 Hz, flattened per channel"
    agent = "A"

    def __init__(self, decim: int = RAW_DECIM, **kw):
        super().__init__(decim=int(decim), **kw)
        self.decim = int(decim)

    def _downsample(self, X: np.ndarray) -> np.ndarray:
        # Non-overlapping block mean (anti-aliased decimation already happened in
        # preprocessing; a block mean here is a cheap, leakage-free further reduction).
        n = X.shape[-1]
        keep = (n // self.decim) * self.decim
        Xb = X[..., :keep].reshape(X.shape[0], X.shape[1], keep // self.decim, self.decim)
        return Xb.mean(axis=-1)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        ds = self._downsample(X)  # (n, 17, n_down)
        self._n_down = ds.shape[-1]
        return ds.reshape(ds.shape[0], -1)

    def feature_names(self) -> list[str]:
        n_down = getattr(self, "_n_down", 750 // self.decim)
        return [f"ch{c:02d}_t{t:03d}" for c in range(N_CHANNELS) for t in range(n_down)]


# ------------------------------------------------------------------ erp_windows
@register
class ErpWindows(BaseExtractor):
    name = "erp_windows"
    description = "Mean amplitude per channel over each of the 8 ERP windows (a.u.)"
    agent = "A"

    def _transform(self, X: np.ndarray) -> np.ndarray:
        cols = []
        for lo, hi in ERP_WINDOWS.values():
            sl = window_slice(lo, hi, SFREQ)
            cols.append(X[..., sl].mean(axis=-1))  # (n, 17)
        return np.concatenate(cols, axis=1)

    def feature_names(self) -> list[str]:
        return [f"ch{c:02d}_{w}_mean" for w in ERP_WINDOWS for c in range(N_CHANNELS)]


# ------------------------------------------------------------------ hjorth
def _hjorth(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hjorth activity, mobility, complexity along the last axis.

    X shape (n, 17, t) -> three (n, 17) arrays.
    """
    d1 = np.diff(X, axis=-1)
    d2 = np.diff(d1, axis=-1)
    var0 = X.var(axis=-1)
    var1 = d1.var(axis=-1)
    var2 = d2.var(axis=-1)
    eps = 1e-12
    activity = var0
    mobility = np.sqrt(var1 / (var0 + eps))
    mob_d1 = np.sqrt(var2 / (var1 + eps))
    complexity = mob_d1 / (mobility + eps)
    return activity, mobility, complexity


@register
class Hjorth(BaseExtractor):
    name = "hjorth"
    description = "Hjorth activity, mobility, complexity per channel"
    agent = "A"

    def _transform(self, X: np.ndarray) -> np.ndarray:
        act, mob, comp = _hjorth(X.astype(np.float64))
        return np.concatenate([act, mob, comp], axis=1)

    def feature_names(self) -> list[str]:
        return (
            channel_feature_names("hjorth_activity")
            + channel_feature_names("hjorth_mobility")
            + channel_feature_names("hjorth_complexity")
        )


# ------------------------------------------------------------------ time_stats
@register
class TimeStats(BaseExtractor):
    name = "time_stats"
    description = "Peak-to-peak, skew, kurtosis, line length, RMS, zero-cross rate per channel"
    agent = "A"

    def _transform(self, X: np.ndarray) -> np.ndarray:
        Xd = X.astype(np.float64)
        n_t = Xd.shape[-1]
        ptp = Xd.max(axis=-1) - Xd.min(axis=-1)
        sk = skew(Xd, axis=-1, bias=False)
        ku = kurtosis(Xd, axis=-1, bias=False, fisher=True)
        line_len = np.abs(np.diff(Xd, axis=-1)).sum(axis=-1)
        rms = np.sqrt((Xd ** 2).mean(axis=-1))
        # zero-crossing rate: fraction of adjacent-sample sign changes
        signs = np.signbit(Xd)
        zcr = np.abs(np.diff(signs.astype(np.int8), axis=-1)).sum(axis=-1) / (n_t - 1)
        feats = np.concatenate([ptp, sk, ku, line_len, rms, zcr], axis=1)
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    def feature_names(self) -> list[str]:
        return (
            channel_feature_names("ptp")
            + channel_feature_names("skew")
            + channel_feature_names("kurtosis")
            + channel_feature_names("line_length")
            + channel_feature_names("rms")
            + channel_feature_names("zero_cross_rate")
        )
