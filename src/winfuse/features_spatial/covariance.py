"""Covariance / Riemannian spatial features (Agent B).

Extractors
----------
cov_tangent      : OAS covariance -> Riemannian tangent space (reference = train mean).
cov_vec          : half-vectorised OAS covariance (Euclidean baseline, measures the
                   value of the Riemannian step).
log_variance     : log of per-channel variance (log of the covariance diagonal).
cov_tangent_ea   : Euclidean-alignment-whitened covariance -> tangent space. Each
                   subject's covariances are whitened by that subject's own reference
                   mean so the feature distributions line up ACROSS subjects. Subjects
                   saw DIFFERENT stimulus sets, so this aligns covariance statistics,
                   NOT paired trials.

All means/filters are fitted on TRAIN epochs only.
"""
from __future__ import annotations

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_covariance

from winfuse.common.config import N_CHANNELS
from winfuse.common.feature_api import BaseExtractor, register

from ._util import COV_ESTIMATOR, tri_names, vech

_N_TRI = N_CHANNELS * (N_CHANNELS + 1) // 2  # 153


def _covs(X: np.ndarray) -> np.ndarray:
    """OAS covariance of each trial. X: (n, 17, T) -> (n, 17, 17) float64 (SPD)."""
    # pyriemann wants (n_trials, n_channels, n_times); returns SPD matrices.
    return Covariances(estimator=COV_ESTIMATOR).transform(X).astype(np.float64)


@register
class CovTangent(BaseExtractor):
    name = "cov_tangent"
    description = (
        "OAS channel covariance projected to the Riemannian tangent space at the "
        "train-set geometric mean; 153 scale-invariant spatial features."
    )
    agent = "B"

    def __init__(self, **kw):
        super().__init__(**kw)
        self._ts: TangentSpace | None = None
        self.params.setdefault("estimator", COV_ESTIMATOR)
        self.params.setdefault("metric", "riemann")

    def _fit(self, X: np.ndarray) -> None:
        C = _covs(X)
        self._ts = TangentSpace(metric="riemann")
        self._ts.fit(C)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        C = _covs(X)
        return self._ts.transform(C).astype(np.float32)

    def feature_names(self) -> list[str]:
        return [f"ts_{i:03d}" for i in range(_N_TRI)]


@register
class CovVec(BaseExtractor):
    name = "cov_vec"
    description = (
        "Half-vectorised OAS channel covariance (lower triangle incl. diagonal); "
        "153 Euclidean spatial features. Baseline for the Riemannian tangent space."
    )
    agent = "B"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.params.setdefault("estimator", COV_ESTIMATOR)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return vech(_covs(X))

    def feature_names(self) -> list[str]:
        return tri_names("cov")


@register
class LogVariance(BaseExtractor):
    name = "log_variance"
    description = (
        "log of per-channel variance (log diagonal of the OAS covariance); "
        "17 features. Simplest amplitude-scaled spatial baseline."
    )
    agent = "B"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.params.setdefault("estimator", COV_ESTIMATOR)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        C = _covs(X)
        diag = np.einsum("nii->ni", C)  # (n, 17)
        return np.log(np.maximum(diag, 1e-12)).astype(np.float32)

    def feature_names(self) -> list[str]:
        return [f"logvar_c{c:02d}" for c in range(N_CHANNELS)]


@register
class CovTangentEA(BaseExtractor):
    name = "cov_tangent_ea"
    description = (
        "Euclidean-alignment: whiten each trial's covariance by the inverse square "
        "root of this subject's train reference mean, then project to the tangent "
        "space at the identity. Makes feature distributions comparable ACROSS "
        "subjects (who saw different stimulus sets -> aligned on covariance "
        "statistics, not paired trials). 153 features."
    )
    agent = "B"

    def __init__(self, **kw):
        super().__init__(**kw)
        self._P: np.ndarray | None = None  # inverse sqrt of reference mean
        self._ts: TangentSpace | None = None
        self.params.setdefault("estimator", COV_ESTIMATOR)
        self.params.setdefault("alignment", "euclidean (Riemannian-mean whitening)")

    def _fit(self, X: np.ndarray) -> None:
        C = _covs(X)
        # Reference = Riemannian geometric mean of this subject's TRAIN covariances.
        Cref = mean_covariance(C, metric="riemann")
        # Inverse square root via eigendecomposition (Cref is SPD).
        w, V = np.linalg.eigh(Cref)
        w = np.maximum(w, 1e-12)
        self._P = (V * (1.0 / np.sqrt(w))) @ V.T  # Cref^{-1/2}
        Caligned = self._P @ C @ self._P  # (n,17,17); mean now ~ identity
        self._ts = TangentSpace(metric="riemann")
        self._ts.fit(Caligned)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        C = _covs(X)
        Caligned = self._P @ C @ self._P
        return self._ts.transform(Caligned).astype(np.float32)

    def feature_names(self) -> list[str]:
        return [f"ea_ts_{i:03d}" for i in range(_N_TRI)]
