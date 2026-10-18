"""Spectral extractors (Agent A).

* band_power     — absolute Welch PSD band power per channel (amplitude-scaled)
* rel_band_power — relative band power (scale-invariant) + spectral entropy,
                   spectral edge frequency, and 1/f spectral slope per channel
"""
from __future__ import annotations

import numpy as np

from winfuse.common.config import BANDS, N_CHANNELS
from winfuse.common.feature_api import BaseExtractor, channel_feature_names, register

from .dsp import WELCH_NOVERLAP, WELCH_NPERSEG, all_band_powers, band_mask, welch_psd

# Spectral edge: frequency below which this fraction of in-band power lies.
SPECTRAL_EDGE_PCT = 0.95
# 1/f slope is fit on log-log PSD over this frequency range (avoids the DC/0.5 Hz
# edge and the band-pass roll-off near 40 Hz).
SLOPE_FMIN = 2.0
SLOPE_FMAX = 35.0


@register
class BandPower(BaseExtractor):
    name = "band_power"
    description = "Welch PSD absolute band power per channel (5 bands x 17 ch), log10"
    agent = "A"

    def __init__(self, **kw):
        super().__init__(nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP,
                         bands=list(BANDS), **kw)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        freqs, psd = welch_psd(X)
        powers = all_band_powers(freqs, psd)  # {band: (n, 17)}
        # log10 to compress the dynamic range; +eps guards log(0).
        cols = [np.log10(powers[b] + 1e-12) for b in BANDS]
        return np.concatenate(cols, axis=1)

    def feature_names(self) -> list[str]:
        return [f"ch{c:02d}_{b}_logpow" for b in BANDS for c in range(N_CHANNELS)]


@register
class RelBandPower(BaseExtractor):
    name = "rel_band_power"
    description = ("Relative band power (scale-invariant) + spectral entropy, "
                   "95% edge freq, and 1/f slope per channel")
    agent = "A"

    def __init__(self, **kw):
        super().__init__(nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP,
                         bands=list(BANDS), spectral_edge_pct=SPECTRAL_EDGE_PCT,
                         slope_fmin=SLOPE_FMIN, slope_fmax=SLOPE_FMAX, **kw)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        freqs, psd = welch_psd(X)  # psd (n, 17, F)
        eps = 1e-12
        powers = all_band_powers(freqs, psd)  # {band: (n, 17)}
        total = sum(powers.values()) + eps  # (n, 17)
        rel = [powers[b] / total for b in BANDS]  # scale-invariant

        # Spectral entropy over the full (band-passed) spectrum, normalised to [0,1].
        p_norm = psd / (psd.sum(axis=-1, keepdims=True) + eps)
        ent = -(p_norm * np.log(p_norm + eps)).sum(axis=-1)
        ent = ent / np.log(psd.shape[-1])  # (n, 17)

        # Spectral edge frequency: freq below which SPECTRAL_EDGE_PCT of power lies.
        csum = np.cumsum(psd, axis=-1)
        csum = csum / (csum[..., -1:] + eps)
        idx = (csum >= SPECTRAL_EDGE_PCT).argmax(axis=-1)  # (n, 17)
        edge = freqs[idx]

        # 1/f slope: linear fit of log10(psd) vs log10(freq) over [fmin, fmax].
        m = band_mask(freqs, SLOPE_FMIN, SLOPE_FMAX)
        lf = np.log10(freqs[m])
        lf = lf - lf.mean()
        denom = float((lf ** 2).sum())
        lp = np.log10(psd[..., m] + eps)  # (n, 17, k)
        slope = (lp * lf).sum(axis=-1) / denom  # (n, 17)

        feats = np.concatenate(rel + [ent, edge, slope], axis=1)
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for b in BANDS:
            names += [f"ch{c:02d}_{b}_relpow" for c in range(N_CHANNELS)]
        names += channel_feature_names("spectral_entropy")
        names += channel_feature_names("spectral_edge95")
        names += channel_feature_names("spectral_slope")
        return names
