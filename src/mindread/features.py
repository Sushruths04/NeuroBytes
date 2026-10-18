"""Video feature cache: one npz per feature set in outputs/mindread_features/<name>.npz with keys
ids (N,), feats (N, D), [feats_t (N, T, D)], model (str), revision (str)."""
from __future__ import annotations

import os

import numpy as np

CACHE = os.environ.get('OURS', os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'mindread_features')))


def frame_picks(n_frames: int, n: int = 8) -> np.ndarray:
    """Organizer-compatible: n frames evenly spaced from first to last."""
    return np.linspace(0, max(n_frames - 1, 0), n).astype(int)


def save(name: str, ids, feats, model: str, revision: str = '', feats_t=None) -> str:
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f'{name}.npz')
    kw = dict(ids=np.asarray(ids), feats=np.asarray(feats, dtype=np.float32), model=model, revision=revision)
    if feats_t is not None:
        kw['feats_t'] = np.asarray(feats_t, dtype=np.float32)
    np.savez(p, **kw)
    return p


def load(name: str, key: str = 'feats') -> dict:
    z = np.load(os.path.join(CACHE, f'{name}.npz'), allow_pickle=False)
    return {int(i): v for i, v in zip(z['ids'], z[key])}


def matrix(table: dict, ids) -> np.ndarray:
    return np.stack([table[int(i)] for i in ids]).reshape(len(ids), -1)
