"""Shared helpers for Agent B's spatial feature extractors.

Nothing here fits on test data. Covariance estimation uses OAS shrinkage by
default (documented in decisions/B.md): with 17 channels and 750 samples the
sample covariance is already well-conditioned (ratio ~44 samples/channel), but
OAS gives a small, analytically-chosen shrinkage that guarantees SPD matrices
for the Riemannian metric without a hand-tuned regulariser.
"""
from __future__ import annotations

import resource
import sys
import time
from contextlib import contextmanager

import numpy as np

from winfuse.common.config import N_CHANNELS

# OAS is the default covariance estimator across the spatial extractors.
COV_ESTIMATOR = "oas"


def peak_rss_mb() -> float:
    """Peak resident set size in MB.

    `resource.ru_maxrss` is bytes on macOS/Darwin and kilobytes on Linux.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


@contextmanager
def timer():
    """Context manager yielding a 1-element list that receives elapsed seconds."""
    out = [0.0]
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        out[0] = time.perf_counter() - t0


def tri_names(prefix: str, n: int = N_CHANNELS) -> list[str]:
    """Lower-triangular (incl. diagonal) index names: <prefix>_cI_cJ (channel-pair)."""
    names = []
    for i in range(n):
        for j in range(i + 1):
            names.append(f"{prefix}_c{i:02d}_c{j:02d}")
    return names


def vech(mats: np.ndarray) -> np.ndarray:
    """Half-vectorisation (lower triangle incl. diagonal) of a stack of matrices.

    mats: (n, c, c) -> (n, c*(c+1)/2), float32.
    """
    n, c, _ = mats.shape
    idx = np.tril_indices(c)
    return mats[:, idx[0], idx[1]].astype(np.float32, copy=False)
