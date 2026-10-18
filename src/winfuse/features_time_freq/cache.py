"""Per-subject feature caching for Agent A (time/frequency extractors).

Features are written to ``cache/features/<method>/<sub>.npy`` as float32 arrays of
shape ``(n_trials, n_features)``. Feature names are written alongside as
``<sub>.names.json`` so a cached array can be validated on load.

This helper lives in Agent A's own package (not in ``src/common/``) per the brief.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from winfuse.common.config import CACHE_DIR

FEATURE_CACHE = CACHE_DIR / "features"
FEATURE_CACHE.mkdir(parents=True, exist_ok=True)


def _method_dir(method: str) -> Path:
    d = FEATURE_CACHE / method
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path(method: str, sub: str) -> Path:
    return _method_dir(method) / f"{sub}.npy"


def names_path(method: str, sub: str) -> Path:
    return _method_dir(method) / f"{sub}.names.json"


def save_features(method: str, sub: str, F: np.ndarray, feature_names: list[str]) -> Path:
    """Write features + names to disk. Returns the .npy path."""
    F = np.ascontiguousarray(np.asarray(F, dtype=np.float32))
    if F.ndim != 2:
        raise ValueError(f"expected 2-D features, got {F.shape}")
    if F.shape[1] != len(feature_names):
        raise ValueError(
            f"{method}/{sub}: {F.shape[1]} columns but {len(feature_names)} names"
        )
    p = cache_path(method, sub)
    np.save(p, F)
    names_path(method, sub).write_text(json.dumps(list(feature_names)))
    return p


def load_features(method: str, sub: str) -> tuple[np.ndarray, list[str]]:
    """Load cached features + names. Raises FileNotFoundError if absent."""
    p = cache_path(method, sub)
    np_names = names_path(method, sub)
    if not p.exists() or not np_names.exists():
        raise FileNotFoundError(f"no cache for {method}/{sub} at {p}")
    F = np.ascontiguousarray(np.load(p).astype(np.float32, copy=False))
    names = json.loads(np_names.read_text())
    return F, names


def has_cache(method: str, sub: str) -> bool:
    return cache_path(method, sub).exists() and names_path(method, sub).exists()


def cache_size_bytes(method: str, sub: str) -> int:
    p = cache_path(method, sub)
    return p.stat().st_size if p.exists() else 0
