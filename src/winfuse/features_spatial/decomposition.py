"""Spatial decomposition features (Agent B): PCA and ICA.

spatial_pca    : PCA over the 17-channel space (fit on train, pooled over trials x
                 time). Components are spatial patterns; features per trial are the
                 log-variance of each component's time course + its mean absolute
                 amplitude. n_components chosen to explain >=99% variance (<=17).
ica_components : MNE FastICA (extended infomax unavailable without picard; FastICA
                 is deterministic with a fixed seed) over channels, fit on train.
                 Features per trial = log-variance of each IC time course. Number of
                 ICs = PCA n_components (whitening rank), capped.

Fit on TRAIN epochs only. Component time courses are summarised per trial so the
feature dimension is fixed and small.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA, FastICA

from winfuse.common.config import N_CHANNELS, SEED
from winfuse.common.feature_api import BaseExtractor, register

_PCA_VAR_TARGET = 0.99


def _stack_ct(X: np.ndarray) -> np.ndarray:
    """(n, c, t) -> (n*t, c) samples-by-channels for component fitting."""
    n, c, t = X.shape
    return np.transpose(X, (0, 2, 1)).reshape(n * t, c)


@register
class SpatialPCA(BaseExtractor):
    name = "spatial_pca"
    description = (
        "Spatial PCA over the 17 channels (fit on train, pooled over trials x time, "
        ">=99% variance). Features per trial: log-variance + mean-abs amplitude of "
        "each component's time course."
    )
    agent = "B"

    def __init__(self, var_target: float = _PCA_VAR_TARGET, **kw):
        super().__init__(**kw)
        self.var_target = float(var_target)
        self._pca: PCA | None = None
        self._k: int | None = None
        self.params.setdefault("var_target", self.var_target)
        self.params.setdefault("seed", SEED)

    def _fit(self, X: np.ndarray) -> None:
        S = _stack_ct(X)
        full = PCA(n_components=N_CHANNELS, random_state=SEED).fit(S)
        cum = np.cumsum(full.explained_variance_ratio_)
        k = int(np.searchsorted(cum, self.var_target) + 1)
        self._k = max(1, min(k, N_CHANNELS))
        self._pca = PCA(n_components=self._k, random_state=SEED).fit(S)
        self.params["n_components"] = self._k

    def _transform(self, X: np.ndarray) -> np.ndarray:
        n, c, t = X.shape
        S = _stack_ct(X)
        Y = self._pca.transform(S).reshape(n, t, self._k)  # (n, t, k)
        logvar = np.log(np.maximum(Y.var(axis=1), 1e-12))   # (n, k)
        meanabs = np.abs(Y).mean(axis=1)                     # (n, k)
        return np.concatenate([logvar, meanabs], axis=1).astype(np.float32)

    def feature_names(self) -> list[str]:
        k = self._k if self._k is not None else N_CHANNELS
        return (
            [f"pca_logvar_comp{i:02d}" for i in range(k)]
            + [f"pca_meanabs_comp{i:02d}" for i in range(k)]
        )


@register
class ICAComponents(BaseExtractor):
    name = "ica_components"
    description = (
        "FastICA spatial components over the 17 channels (fit on train, fixed seed, "
        "max_iter=500). Features per trial = log-variance of each IC time course. "
        "n_components matches the PCA 99%-variance rank, capped at 15."
    )
    agent = "B"

    def __init__(self, n_components: int | None = None, max_iter: int = 500, **kw):
        super().__init__(**kw)
        self._n = n_components
        self.max_iter = int(max_iter)
        self._ica: FastICA | None = None
        self.params.setdefault("max_iter", self.max_iter)
        self.params.setdefault("algorithm", "FastICA (parallel, logcosh)")
        self.params.setdefault("seed", SEED)

    def _fit(self, X: np.ndarray) -> None:
        S = _stack_ct(X)
        if self._n is None:
            full = PCA(n_components=N_CHANNELS, random_state=SEED).fit(S)
            cum = np.cumsum(full.explained_variance_ratio_)
            k = int(np.searchsorted(cum, _PCA_VAR_TARGET) + 1)
            self._n = max(2, min(k, 15))
        self._ica = FastICA(
            n_components=self._n,
            max_iter=self.max_iter,
            whiten="unit-variance",
            random_state=SEED,
        )
        self._ica.fit(S)
        self.params["n_components"] = self._n

    def _transform(self, X: np.ndarray) -> np.ndarray:
        n, c, t = X.shape
        S = _stack_ct(X)
        Y = self._ica.transform(S).reshape(n, t, self._n)  # (n, t, k)
        return np.log(np.maximum(Y.var(axis=1), 1e-12)).astype(np.float32)

    def feature_names(self) -> list[str]:
        k = self._n if self._n is not None else 2
        return [f"ica_logvar_comp{i:02d}" for i in range(k)]
