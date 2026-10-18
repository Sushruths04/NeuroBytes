"""Ranking metrics and submission formatting."""
from __future__ import annotations

import numpy as np

from .data import TEST_IDS


def ranks_of_truth(scores: np.ndarray, truth_cols: np.ndarray) -> np.ndarray:
    """scores (n_rec, n_cand), higher = more likely; returns 1-based rank of the true candidate."""
    order = np.argsort(-scores, axis=1, kind='stable')
    pos = np.empty_like(order)
    pos[np.arange(len(order))[:, None], order] = np.arange(order.shape[1])[None, :]
    return pos[np.arange(len(scores)), truth_cols] + 1


def metrics(r: np.ndarray) -> dict:
    r = np.asarray(r)
    return dict(top1=float((r == 1).mean()), top5=float((r <= 5).mean()),
                mean_rank=float(r.mean()), median_rank=float(np.median(r)), mrr=float((1.0 / r).mean()))


def to_ranked_ids(scores: np.ndarray, cand_ids: np.ndarray = TEST_IDS) -> np.ndarray:
    """Rows stay in the ORIGINAL recording order; returns int64 (n_rec, 102) video IDs, best first."""
    return np.asarray(cand_ids)[np.argsort(-scores, axis=1, kind='stable')].astype(np.int64)


def validate_submission(arr: np.ndarray) -> None:
    if arr.shape != (102, 102):
        raise ValueError(f'shape {arr.shape} != (102, 102)')
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f'dtype {arr.dtype} is not integer')
    target = np.sort(TEST_IDS)
    for i, row in enumerate(arr):
        if not np.array_equal(np.sort(row), target):
            raise ValueError(f'row {i} is not a permutation of 1001..1102')
