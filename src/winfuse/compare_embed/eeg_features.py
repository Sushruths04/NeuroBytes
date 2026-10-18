"""EEG feature matrices and embedding resolution, aligned by video ID.

This module is the single place that:
  * turns a Phase-1 extractor name into an (n_trials, n_features) matrix keyed by
    the subject's training video IDs, via get_extractor() fitted on TRAIN epochs;
  * exposes the time-resolved preprocessed array (n, 17, 750) @250 Hz and the
    100 ms sliding-window slices used by the latency analysis;
  * resolves a model name to its (train) embedding matrix for a requested set of
    video IDs, coping with the two cache-naming schemes in play
    (baseline_alexnet uses baseline_train/baseline_test; V-A/V-B models use
    pooled / layer_* over all 1102 ids).

Nothing here reads tst/ or tst_stim/.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from winfuse.common import embed_cache as ec
from winfuse.common import preprocess as pp
from winfuse.common.config import SFREQ
from winfuse.common.feature_api import get_extractor

# Phase-1 feature sets to use (cov_tangent_ea and ersp_trialmean dropped as proven
# per-column-affine-identical duplicates in results/_redundancy/metrics.json).
EEG_FEATURE_SETS = (
    "raw_signal",
    "erp_windows",
    "band_power",
    "rel_band_power",
    "tf_band_power_reduced",
    "xdawn_evoked",
    "cov_tangent",
)

# Latency analysis sliding window
LATENCY_WIN_S = 0.100       # 100 ms
LATENCY_STEP_S = 0.100      # non-overlapping 100 ms steps -> 30 windows over 3 s


# --------------------------------------------------------------------------- EEG
def eeg_feature_matrix(subject: str, feature_set: str) -> tuple[np.ndarray, list[int]]:
    """(n_trials, n_features) float32 for one subject + extractor, train only.

    Returns (X, ids) where ids are the subject's training video IDs in the same row
    order. The extractor is fit on the subject's preprocessed TRAIN epochs; nothing
    test is ever touched.
    """
    Xep, ids = pp.preprocessed_train(subject)          # (n, 17, 750), ids
    ext = get_extractor(feature_set)
    ext.fit(Xep)                                       # BaseExtractor.fit(X_train) only
    F = np.asarray(ext.transform(Xep), dtype=np.float32)
    if F.ndim != 2 or F.shape[0] != Xep.shape[0]:
        raise ValueError(f"{feature_set}: transform returned {F.shape}, "
                         f"expected (n_trials, n_features) with n_trials={Xep.shape[0]}")
    return F, list(ids)


def preprocessed_timeresolved(subject: str) -> tuple[np.ndarray, list[int]]:
    """The preprocessed (n, 17, 750) @250 Hz array itself, for latency analysis."""
    X, ids = pp.preprocessed_train(subject)
    return np.asarray(X, dtype=np.float32), list(ids)


def latency_windows(n_samples: int = 750, sfreq: float = SFREQ,
                    win_s: float = LATENCY_WIN_S,
                    step_s: float = LATENCY_STEP_S) -> list[tuple[float, float, slice]]:
    """List of (t0, t1, slice) for non-overlapping 100 ms windows over the epoch."""
    win = int(round(win_s * sfreq))
    step = int(round(step_s * sfreq))
    out = []
    i = 0
    while i + win <= n_samples:
        t0 = i / sfreq
        t1 = (i + win) / sfreq
        out.append((t0, t1, slice(i, i + win)))
        i += step
    return out


def window_feature_matrix(Xtr: np.ndarray, sl: slice) -> np.ndarray:
    """Flatten a time window of the preprocessed array to (n_trials, 17*win)."""
    seg = Xtr[:, :, sl]
    return seg.reshape(seg.shape[0], -1).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- embeddings
def _train_variant_candidates(model: str) -> list[str]:
    """Variant names that could hold TRAIN-video embeddings, best-first."""
    avail = ec.available(model)
    # explicit train variant first, then generic pooled, then layer_* / timeseries
    ordered = []
    for name in ("baseline_train", "pooled", "mean", "features"):
        if name in avail:
            ordered.append(name)
    for v in avail:
        if v not in ordered and v not in ("baseline_test", "ids", "times"):
            ordered.append(v)
    return ordered


def _variant_ids(model: str, variant: str) -> list[int] | None:
    """Video IDs for a consolidated variant, in its row order.

    The frozen embed_cache stores a single shared ids.npy per model, which the LAST
    consolidate() call overwrites. When a model has multiple variants over different
    ID sets (e.g. baseline_train 1..1000 vs baseline_test 1001..1102), that shared
    file is therefore only correct for one of them. We instead reconstruct the IDs
    from the per-variant _parts directory via done_ids(), which returns the sorted
    part IDs -- exactly the ascending-ID row order consolidate() writes. We confirm
    the count matches the consolidated array before trusting it, else fall back to
    the shared ids.npy.
    """
    try:
        X, shared_ids = ec.load(model, variant)
    except Exception:
        return None
    part_ids = ec.done_ids(model, variant)
    if len(part_ids) == X.shape[0]:
        return [int(v) for v in part_ids]
    if len(shared_ids) == X.shape[0]:
        return [int(v) for v in shared_ids]
    return None


def resolve_embedding(model: str, wanted_ids: Iterable[int],
                      variant: str | None = None) -> tuple[np.ndarray, list[int], str]:
    """Embedding rows for `wanted_ids` (training IDs), as (X, ids_found, variant_used).

    If `variant` is given it is used directly. Otherwise the first variant that
    actually contains at least one of the wanted IDs is chosen. Pooled embeddings
    that are >2-D (e.g. accidental (n,1,D)) are flattened to (n, D).
    """
    wanted = [int(v) for v in wanted_ids]
    cands = [variant] if variant else _train_variant_candidates(model)
    last_err = None
    for v in cands:
        if v is None:
            continue
        vids = _variant_ids(model, v)
        if vids is None:
            continue
        try:
            X, _ = ec.load(model, v)
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
        X = np.asarray(X, dtype=np.float32)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        pos = {vid: i for i, vid in enumerate(vids)}
        keep = [vid for vid in wanted if vid in pos]
        if not keep:
            continue
        idx = [pos[vid] for vid in keep]
        return X[idx], keep, v
    raise FileNotFoundError(
        f"{model}: no variant yields embeddings for the requested ids "
        f"(tried {cands}); last error: {last_err!r}"
    )


def available_models(include_synthetic: bool = True) -> list[str]:
    """Models with at least one consolidated variant on disk."""
    out = [m for m in ec.list_models() if ec.available(m)]
    return out


def layer_variants(model: str) -> list[str]:
    """Layerwise variants of a model (names starting with 'layer_'), sorted."""
    return sorted(v for v in ec.available(model) if v.startswith("layer_"))


def timeseries_variant(model: str) -> str | None:
    """A time-resolved variant name if one exists (for latency vs audio)."""
    for v in ec.available(model):
        if v in ("timeseries", "timeresolved") or v.startswith("timeseries"):
            return v
    return None


def align_eeg_embedding(F: np.ndarray, eeg_ids: list[int],
                        model: str, variant: str | None = None,
                        ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Align an EEG feature matrix and a model's embeddings on common video IDs.

    Returns (F_aligned, E_aligned, common_ids) with rows in the same video order.
    """
    E_all, emb_ids, _ = resolve_embedding(model, eeg_ids, variant)
    emb_pos = {int(v): i for i, v in enumerate(emb_ids)}
    eeg_pos = {int(v): i for i, v in enumerate(eeg_ids)}
    common = [v for v in eeg_ids if v in emb_pos]
    if not common:
        raise ValueError(f"{model}: no common video IDs with EEG")
    fi = [eeg_pos[v] for v in common]
    ei = [emb_pos[v] for v in common]
    return (np.asarray(F)[fi].astype(np.float32),
            np.asarray(E_all)[ei].astype(np.float32),
            common)
