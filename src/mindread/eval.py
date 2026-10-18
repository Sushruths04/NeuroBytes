"""Two validation signals: CV on training videos (all 6 subjects) and logged sub-01..04 official-test validation."""
from __future__ import annotations

import csv, datetime, hashlib, json, os

import numpy as np

from .data import load_val_labels
from .scoring import metrics, ranks_of_truth

VAL_LOG = os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'dev_test_validation_log.csv')


def video_folds(ids: np.ndarray, n_folds: int = 5, seed: int = 0, groups: dict | None = None):
    """Yield (train_idx, held_idx). `groups` maps video id -> group id so near-duplicates stay together."""
    ids = np.asarray(ids)
    g = np.array([groups.get(int(i), int(i)) if groups else int(i) for i in ids])
    ug = np.unique(g)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(ug, rng.permutation(len(ug)) % n_folds))
    f = np.array([fold_of[x] for x in g])
    for k in range(n_folds):
        yield np.where(f != k)[0], np.where(f == k)[0]


def retrieval_draws(held_idx: np.ndarray, n_cand: int = 102, n_draws: int = 20, seed: int = 0):
    """Random 102-candidate sets from held-out training videos (mimics the official task)."""
    rng = np.random.default_rng(seed)
    for _ in range(n_draws):
        yield rng.choice(held_idx, size=min(n_cand, len(held_idx)), replace=False)


def dev_validation(root: str, subject: int, scores: np.ndarray, config: dict, exp_id: str) -> dict:
    """Score sub-01..04 official test predictions (PDF: allowed for validation and model selection).

    Every call is logged, so the number of looks is auditable. A change is promoted only if it also
    improves the training-video CV benchmark (see docs/src/action_plan.md, promotion rule).
    """
    os.makedirs(os.path.dirname(VAL_LOG), exist_ok=True)
    cfg_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]
    gt = load_val_labels(root, subject)
    m = metrics(ranks_of_truth(scores, gt - 1001))
    new = not os.path.exists(VAL_LOG)
    with open(VAL_LOG, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['time', 'exp_id', 'config_hash', 'subject', 'top1', 'top5', 'mean_rank', 'mrr'])
        if new:
            w.writeheader()
        w.writerow(dict(time=datetime.datetime.now().isoformat(timespec='seconds'), exp_id=exp_id, config_hash=cfg_hash,
                        subject=subject, top1=m['top1'], top5=m['top5'], mean_rank=m['mean_rank'], mrr=m['mrr']))
    return m
