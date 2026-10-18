"""Time-frequency extractors (Agent A).

* tf_band_power          — full Morlet band power, 17 ch x 5 bands x 23 windows
* tf_band_power_reduced  — same, but time-pooled into 4 coarse epochs x 5 bands x 17 ch
* ersp_trialmean         — tf power expressed in dB relative to the TRAIN grand-average
                           power (fitted parameter learned in _fit), then time-pooled
"""
from __future__ import annotations

import numpy as np

from winfuse.common.config import BANDS, N_CHANNELS
from winfuse.common.feature_api import BaseExtractor, register

from .tf_core import tf_band_power as _tf_band_power

# Number of coarse temporal epochs for the reduced / ERSP variants.
N_COARSE = 4


def _coarse_pool(power: np.ndarray, n_coarse: int = N_COARSE) -> np.ndarray:
    """Average TF power (n, 17, bands, win) into n_coarse temporal chunks.

    Returns (n, 17, bands, n_coarse).
    """
    n_win = power.shape[-1]
    bounds = np.linspace(0, n_win, n_coarse + 1).astype(int)
    chunks = [power[..., bounds[i]:bounds[i + 1]].mean(axis=-1) for i in range(n_coarse)]
    return np.stack(chunks, axis=-1)


@register
class TfBandPower(BaseExtractor):
    name = "tf_band_power"
    description = "Morlet time-resolved log band power, 17ch x 5 bands x 23 windows"
    agent = "A"

    def _transform(self, X: np.ndarray) -> np.ndarray:
        power, bands, edges = _tf_band_power(X)
        self._bands = bands
        self._n_win = len(edges)
        return power.reshape(power.shape[0], -1)

    def feature_names(self) -> list[str]:
        bands = getattr(self, "_bands", list(BANDS))
        n_win = getattr(self, "_n_win", 23)
        return [
            f"ch{c:02d}_{b}_w{w:02d}_logpow"
            for c in range(N_CHANNELS)
            for b in bands
            for w in range(n_win)
        ]


@register
class TfBandPowerReduced(BaseExtractor):
    name = "tf_band_power_reduced"
    description = "Morlet band power time-pooled to 4 coarse epochs, 17ch x 5 bands x 4"
    agent = "A"

    def __init__(self, n_coarse: int = N_COARSE, **kw):
        super().__init__(n_coarse=int(n_coarse), **kw)
        self.n_coarse = int(n_coarse)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        power, bands, _ = _tf_band_power(X)
        self._bands = bands
        pooled = _coarse_pool(power, self.n_coarse)  # (n, 17, bands, n_coarse)
        return pooled.reshape(pooled.shape[0], -1)

    def feature_names(self) -> list[str]:
        bands = getattr(self, "_bands", list(BANDS))
        return [
            f"ch{c:02d}_{b}_epoch{e}_logpow"
            for c in range(N_CHANNELS)
            for b in bands
            for e in range(self.n_coarse)
        ]


@register
class ErspTrialMean(BaseExtractor):
    name = "ersp_trialmean"
    description = ("Time-resolved power in dB relative to the TRAIN grand-average "
                   "power (no baseline exists), time-pooled to 4 coarse epochs")
    agent = "A"

    def __init__(self, n_coarse: int = N_COARSE, **kw):
        super().__init__(n_coarse=int(n_coarse), baseline="train_grand_average_dB", **kw)
        self.n_coarse = int(n_coarse)
        self.grand_log_mean_: np.ndarray | None = None  # (17, bands, win)

    def _fit(self, X: np.ndarray) -> None:
        # Learn the grand-average LOG power across TRAIN trials. This is the fitted
        # parameter; because no pre-stimulus baseline exists, the across-trial mean
        # power is used as the reference. Fitted on train only -> reused in transform.
        power, bands, edges = _tf_band_power(X)  # (n, 17, bands, win), already log10
        self._bands = bands
        self._n_win = len(edges)
        self.grand_log_mean_ = power.mean(axis=0).astype(np.float32)  # (17, bands, win)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.grand_log_mean_ is None:
            raise RuntimeError("ersp_trialmean: fit() must run before transform()")
        power, bands, _ = _tf_band_power(X)  # log10 power
        # dB relative to train grand average: 10*(log10(P) - log10(Pref)).
        ersp = 10.0 * (power - self.grand_log_mean_[None, ...])
        pooled = _coarse_pool(ersp, self.n_coarse)
        return pooled.reshape(pooled.shape[0], -1).astype(np.float32)

    def feature_names(self) -> list[str]:
        bands = getattr(self, "_bands", list(BANDS))
        return [
            f"ch{c:02d}_{b}_epoch{e}_dB"
            for c in range(N_CHANNELS)
            for b in bands
            for e in range(self.n_coarse)
        ]
