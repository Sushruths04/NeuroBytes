"""Small, regularized score fusion fitted ONLY on inner held-out predictions.

The caller owns split provenance. Outer evaluation scores/labels must never be
passed to fit(). Every component is standardized across the same candidates.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def row_z(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError('scores must be a finite query by candidate matrix')
    return (scores - scores.mean(1, keepdims=True)) / (scores.std(1, keepdims=True) + 1e-12)


class ResidualFusion:
    """Anchor + nonnegative bounded residual scorers; shrinkage and optional
    adjacent-window smoothness discourage fitting noise in a small calibration set.

    Coefficients minimize mean candidate cross entropy + l2*||w||²/2
    + smooth*sum(diff(w)²)/2. Fixed anchor weight 1; no outer-set tuning.
    """

    def __init__(self, names, l2=1.0, smooth=0.0, groups=None):
        self.names = tuple(names)
        self.l2, self.smooth = float(l2), float(smooth)
        # groups: one label per name; smoothness only links adjacent names with the same label
        g = list(groups) if groups is not None else [0] * len(self.names)
        if len(g) != len(self.names):
            raise ValueError('groups must give one label per name')
        self.link = np.array([a == b for a, b in zip(g[:-1], g[1:])], dtype=float)
        if not self.names or l2 < 0 or smooth < 0:
            raise ValueError('need streams and nonnegative penalties')

    def fit(self, components, truth):
        anchor = row_z(components['baseline'])
        x = np.stack([row_z(components[n]) for n in self.names], -1)
        truth = np.asarray(truth)
        if (truth.shape != (len(anchor),) or not np.issubdtype(truth.dtype, np.integer)
                or np.any(truth < 0) or np.any(truth >= anchor.shape[1])):
            raise ValueError('invalid calibration truth indices')
        rows = np.arange(len(truth))

        def objective(w):
            logits = anchor + x @ w
            log_z = logsumexp(logits, axis=1)
            loss = np.mean(log_z - logits[rows, truth]) + self.l2 * (w @ w) / 2
            p = np.exp(logits - log_z[:, None])
            p[rows, truth] -= 1
            grad = np.einsum('nc,nck->k', p, x) / len(truth) + self.l2 * w
            diff = np.diff(w) * self.link
            loss += self.smooth * (diff @ diff) / 2
            grad[:-1] -= self.smooth * diff
            grad[1:] += self.smooth * diff
            return loss, grad

        result = minimize(objective, np.zeros(len(self.names)), jac=True,
                          method='L-BFGS-B', bounds=[(0, 1)] * len(self.names))
        if not result.success:
            raise RuntimeError(f'fusion optimization failed: {result.message}')
        self.weights = result.x
        self.loss = float(result.fun)
        return self

    def scores(self, components):
        return row_z(components['baseline']) + sum(
            w * row_z(components[n]) for n, w in zip(self.names, self.weights))
