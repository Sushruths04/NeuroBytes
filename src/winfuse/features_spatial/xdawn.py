"""xDAWN-style evoked-subspace spatial filtering (Agent B).

The problem: xDAWN and CSP are supervised. Here every trial is a DIFFERENT
stimulus, so there are no class labels to drive a supervised filter. We do NOT
invent pseudo-labels.

What we do instead (logged as a decision): the classic xDAWN signal model is
X = A * E + noise, where E is the evoked (phase-locked, stimulus-independent)
response and A is the spatial mixing. xDAWN finds spatial filters that maximise
the evoked-to-total power ratio. The evoked response E is estimated as the
across-trial average (the grand-average ERP) — this needs NO class labels, only
the fact that t=0 is a common onset. We then solve the generalised eigenproblem

    Cov(evoked) w = lambda Cov(total) w

on TRAIN data and keep the top `n_filters` filters. Features per trial are the
log-variance of each filtered component time course (an evoked-SNR-ranked,
dimensionality-reduced spatial summary).

This is the unsupervised evoked-subspace analogue of xDAWN; it is honest about
the absence of labels. See decisions/B.md.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from winfuse.common.config import N_CHANNELS
from winfuse.common.feature_api import BaseExtractor, register

_DEFAULT_N_FILTERS = 6


@register
class XdawnEvoked(BaseExtractor):
    name = "xdawn_evoked"
    description = (
        "Unsupervised xDAWN-style spatial filters: generalised eigenvectors of "
        "Cov(grand-average ERP) vs Cov(total signal), fitted on train. Features = "
        "log-variance of the top-6 evoked-SNR components per trial. No class labels "
        "used (every trial is a unique stimulus); the evoked subspace is the "
        "across-trial average response."
    )
    agent = "B"

    def __init__(self, n_filters: int = _DEFAULT_N_FILTERS, **kw):
        super().__init__(**kw)
        self.n_filters = int(n_filters)
        self._W: np.ndarray | None = None  # (17, n_filters)
        self.params.setdefault("n_filters", self.n_filters)
        self.params.setdefault("evoked_estimate", "across-trial grand average (no labels)")

    def _fit(self, X: np.ndarray) -> None:
        n, c, t = X.shape
        # Evoked response: average across trials -> (c, t).
        evoked = X.mean(axis=0)
        # Signal (evoked) covariance, scaled like the per-trial total covariance.
        Cs = (evoked @ evoked.T) / t
        # Total covariance: average of per-trial covariances.
        Ct = np.einsum("nct,ndt->cd", X, X) / (n * t)
        # Regularise Ct lightly for a stable generalised eigenproblem.
        Ct = Ct + 1e-6 * np.eye(c, dtype=Ct.dtype)
        # Generalised eigenproblem Cs w = lambda Ct w; eigenvalues ascending.
        w_eig, V = eigh(Cs.astype(np.float64), Ct.astype(np.float64))
        order = np.argsort(w_eig)[::-1]  # descending evoked-SNR
        self._W = V[:, order[: self.n_filters]].astype(np.float32)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        # Apply filters: (n, k, t) = W^T applied per trial.
        Y = np.einsum("ck,nct->nkt", self._W, X)  # (n, k, t)
        var = Y.var(axis=2)  # (n, k)
        return np.log(np.maximum(var, 1e-12)).astype(np.float32)

    def feature_names(self) -> list[str]:
        return [f"xdawn_logvar_comp{i:02d}" for i in range(self.n_filters)]
