"""Linear encoding model: video features -> EEG target, with train-only scaling/PCA and CV ridge."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .data import assert_training_ids


@dataclass
class RidgeEncoder:
    pca: float | int | None = None            # e.g. 0.95 (organizer baseline) or None
    alphas: np.ndarray = field(default_factory=lambda: np.logspace(-3, 6, 20))
    scale_target: bool = True

    def fit(self, ids, F, Y):
        assert_training_ids(ids)
        self.sc1 = StandardScaler().fit(F); Z = self.sc1.transform(F)
        if self.pca is not None:
            self.p = PCA(n_components=self.pca, svd_solver='full').fit(Z); Z = self.p.transform(Z)
            self.sc2 = StandardScaler().fit(Z); Z = self.sc2.transform(Z)
        self.mu, self.sd = (Y.mean(0), Y.std(0) + 1e-9) if self.scale_target else (0.0, 1.0)
        self.r = RidgeCV(alphas=self.alphas).fit(Z, (Y - self.mu) / self.sd)
        return self

    def _z(self, F):
        Z = self.sc1.transform(F)
        if self.pca is not None:
            Z = self.sc2.transform(self.p.transform(Z))
        return Z

    def predict(self, F):
        return self.r.predict(self._z(F)) * self.sd + self.mu


def corr_scores(Y_obs: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Pearson correlation of every observed recording (rows) with every predicted candidate (cols)."""
    a = Y_obs - Y_obs.mean(1, keepdims=True); a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b = Y_pred - Y_pred.mean(1, keepdims=True); b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    return a @ b.T
