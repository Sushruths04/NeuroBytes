"""Embedding cache. FROZEN CONTRACT - sub-agents must not edit this file.

Layout, per model:

    cache/embeddings/<model>/
        _parts/<variant>/<video_id>.npy     per-video partials (resume state)
        <variant>.npy                       consolidated (n_videos, D) or (n_videos, T, D)
        ids.npy                             int32 video IDs, row order of <variant>.npy
        times.npy                           float64 (T,) for time-resolved variants
        manifest.json                        provenance; see `write_manifest`

Resume safety
-------------
Each video is written to its own `_parts/<variant>/<id>.npy` immediately after it is
embedded. A crash therefore loses at most one video. `pending()` returns the IDs that
still need work, so re-running the same command continues rather than restarting.
`consolidate()` stacks the parts into one array in ascending-ID order.

Variants are stored separately because they have different shapes: `pooled` is
(n, D), `timeseries` is (n, T, D), and each layer of `layers` is stored as its own
variant named `layer_<name>`.

Everything is float32 on disk (hard constraint 1).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import CACHE_DIR

EMB_ROOT = CACHE_DIR / "embeddings"
EMB_ROOT.mkdir(parents=True, exist_ok=True)

ALL_VIDEO_IDS = list(range(1, 1001)) + list(range(1001, 1103))   # 1..1000 train, 1001..1102 test


def model_dir(model: str) -> Path:
    d = EMB_ROOT / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def parts_dir(model: str, variant: str) -> Path:
    d = model_dir(model) / "_parts" / variant
    d.mkdir(parents=True, exist_ok=True)
    return d


def part_path(model: str, variant: str, vid: int) -> Path:
    return parts_dir(model, variant) / f"{int(vid):04d}.npy"


# --------------------------------------------------------------------------- write
def save_part(model: str, variant: str, vid: int, arr: np.ndarray) -> Path:
    """Write one video's embedding atomically (tmp + replace), float32."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    if not np.isfinite(a).all():
        raise ValueError(f"{model}/{variant}/{vid}: non-finite values, refusing to cache")
    p = part_path(model, variant, vid)
    # NOTE: the temp name must also end in .npy, because np.save appends .npy to any
    # path that does not. Writing through an open file handle avoids that entirely.
    tmp = p.with_name(p.stem + ".tmp.npy")
    with open(tmp, "wb") as f:
        np.save(f, a)
    tmp.replace(p)
    return p


def have(model: str, variant: str, vid: int) -> bool:
    return part_path(model, variant, vid).exists()


def pending(model: str, variant: str, ids: Iterable[int] | None = None) -> list[int]:
    """Video IDs not yet embedded for this model+variant."""
    ids = list(ids) if ids is not None else ALL_VIDEO_IDS
    return [v for v in ids if not have(model, variant, v)]


def done_ids(model: str, variant: str) -> list[int]:
    d = parts_dir(model, variant)
    return sorted(int(p.stem) for p in d.glob("*.npy"))


# --------------------------------------------------------------------------- read
def load_part(model: str, variant: str, vid: int) -> np.ndarray:
    return np.load(part_path(model, variant, vid))


def consolidate(model: str, variant: str, ids: Iterable[int] | None = None,
                times: np.ndarray | None = None, strict: bool = True) -> dict:
    """Stack per-video parts into <variant>.npy + ids.npy, in ascending-ID order.

    strict=True raises if any requested ID is missing; False consolidates what exists
    and reports the gaps.
    """
    want = sorted(ids) if ids is not None else ALL_VIDEO_IDS
    missing = [v for v in want if not have(model, variant, v)]
    if missing and strict:
        raise FileNotFoundError(
            f"{model}/{variant}: {len(missing)} missing videos, e.g. {missing[:10]}. "
            f"Re-run the extraction (it is resume-safe) or pass strict=False."
        )
    use = [v for v in want if have(model, variant, v)]
    if not use:
        raise FileNotFoundError(f"{model}/{variant}: nothing cached yet")

    first = load_part(model, variant, use[0])
    shape = first.shape
    X = np.empty((len(use), *shape), dtype=np.float32)
    bad_shape = []
    for i, v in enumerate(use):
        a = load_part(model, variant, v)
        if a.shape != shape:
            bad_shape.append((v, a.shape))
            continue
        X[i] = a
    if bad_shape:
        raise ValueError(f"{model}/{variant}: inconsistent part shapes, expected {shape}, "
                         f"got e.g. {bad_shape[:5]}")

    d = model_dir(model)
    np.save(d / f"{variant}.npy", X)
    # Per-variant IDs. A model can hold variants over DIFFERENT id sets (e.g.
    # baseline_alexnet has baseline_train = 1..1000 and baseline_test = 1001..1102),
    # so a single shared ids.npy is clobbered by whichever variant consolidates last.
    # Reported by Agent V-C in questions.md; fixed here.
    np.save(d / f"{variant}.ids.npy", np.asarray(use, dtype=np.int32))
    # Shared ids.npy is still written for backward compatibility with anything that
    # reads it, but <variant>.ids.npy is authoritative and is preferred by load().
    np.save(d / "ids.npy", np.asarray(use, dtype=np.int32))
    if times is not None:
        np.save(d / f"{variant}.times.npy", np.asarray(times, dtype=np.float64))
        np.save(d / "times.npy", np.asarray(times, dtype=np.float64))
    return {"model": model, "variant": variant, "shape": list(X.shape),
            "n_videos": len(use), "n_missing": len(missing),
            "missing_ids": missing[:50], "dtype": "float32",
            "ids_path": str((d / f"{variant}.ids.npy").name),
            "path": str((d / f"{variant}.npy").relative_to(CACHE_DIR.parent))}


def load(model: str, variant: str = "pooled") -> tuple[np.ndarray, np.ndarray]:
    """Load a consolidated variant. Returns (X, ids).

    Prefers the per-variant `<variant>.ids.npy`. Falls back to the shared `ids.npy`,
    and if that length does not match, reconstructs the IDs from the `_parts`
    directory, which is always correct because `consolidate` writes rows in ascending
    part-ID order.
    """
    d = model_dir(model)
    xp = d / f"{variant}.npy"
    if not xp.exists():
        raise FileNotFoundError(f"{xp} not found; run consolidate() for {model}/{variant}")
    X = np.load(xp)

    vip = d / f"{variant}.ids.npy"
    if vip.exists():
        ids = np.load(vip)
        if len(ids) == X.shape[0]:
            return X, ids
    ip = d / "ids.npy"
    if ip.exists():
        ids = np.load(ip)
        if len(ids) == X.shape[0]:
            return X, ids
    recon = np.asarray(done_ids(model, variant), dtype=np.int32)
    if len(recon) == X.shape[0]:
        return X, recon
    raise FileNotFoundError(
        f"{model}/{variant}: cannot determine row IDs. {variant}.npy has {X.shape[0]} rows; "
        f"{variant}.ids.npy {'exists' if vip.exists() else 'missing'}, "
        f"ids.npy {'exists' if ip.exists() else 'missing'}, "
        f"_parts has {len(recon)}. Re-run consolidate()."
    )


def load_times(model: str, variant: str | None = None) -> np.ndarray | None:
    d = model_dir(model)
    if variant:
        p = d / f"{variant}.times.npy"
        if p.exists():
            return np.load(p)
    p = d / "times.npy"
    return np.load(p) if p.exists() else None


def available(model: str) -> list[str]:
    """Consolidated variants available for a model."""
    d = model_dir(model)
    return sorted(p.stem for p in d.glob("*.npy")
                  if p.stem not in ("ids", "times"))


def list_models() -> list[str]:
    return sorted(p.name for p in EMB_ROOT.iterdir() if p.is_dir())


def rows_for_ids(model: str, wanted_ids: Iterable[int],
                 variant: str = "pooled") -> tuple[np.ndarray, list[int]]:
    """Rows of a consolidated variant for specific video IDs, in the order given.

    Returns (X_subset, ids_found). IDs with no cached embedding are skipped and the
    caller can compare `ids_found` against what it asked for.
    """
    X, ids = load(model, variant)
    pos = {int(v): i for i, v in enumerate(ids)}
    want = [int(v) for v in wanted_ids]
    keep = [v for v in want if v in pos]
    idx = [pos[v] for v in keep]
    return X[idx], keep


# --------------------------------------------------------------------------- manifest
def _sha256_file(p: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            if limit and n >= limit:
                break
    return h.hexdigest()


def weights_checksum(paths: Iterable[Path | str], limit_per_file: int | None = None) -> dict:
    """SHA-256 of each weight file, for the model manifest (constraint 2)."""
    out = {}
    for p in paths:
        p = Path(p)
        if p.exists() and p.is_file():
            out[p.name] = {"sha256": _sha256_file(p, limit_per_file),
                           "bytes": p.stat().st_size}
    return out


def hf_revision(repo_id: str) -> dict:
    """Resolve an HF repo's commit sha without downloading weights."""
    try:
        from huggingface_hub import model_info

        i = model_info(repo_id)
        return {"repo_id": repo_id, "sha": i.sha,
                "last_modified": str(getattr(i, "lastModified", None))}
    except Exception as e:  # noqa: BLE001
        return {"repo_id": repo_id, "error": repr(e)}


def _env() -> dict:
    def ver(m):
        try:
            return getattr(__import__(m), "__version__", None)
        except Exception:
            return None

    try:
        g = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        rev = g.stdout.strip() if g.returncode == 0 else None
    except Exception:
        rev = None
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "machine": platform.machine(), "git_rev": rev,
            "numpy": ver("numpy"), "torch": ver("torch"),
            "torchvision": ver("torchvision"), "transformers": ver("transformers"),
            "open_clip": ver("open_clip"), "timm": ver("timm"),
            "librosa": ver("librosa"), "cv2": ver("cv2")}


def write_manifest(model: str, **fields: Any) -> Path:
    """Write cache/embeddings/<model>/manifest.json, merging with any existing one.

    Expected fields (constraint 2 requires the first four):
        source, revision, weights_checksum, n_params,
        device, dtype, frame_sampling, preprocessing,
        wall_clock_s, mps_cpu_check, variants, modality, description,
        silent_ids, substituted_frames, effective_dim
    """
    p = model_dir(model) / "manifest.json"
    base: dict[str, Any] = {}
    if p.exists():
        try:
            base = json.loads(p.read_text())
        except Exception:
            base = {}
    base.update({"model": model, "written": _dt.datetime.now().astimezone()
                 .isoformat(timespec="seconds"), "environment": _env()})
    base.update(_jsonable(fields))
    p.write_text(json.dumps(base, indent=2))
    return p


def read_manifest(model: str) -> dict:
    p = model_dir(model) / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def all_manifests() -> dict[str, dict]:
    return {m: read_manifest(m) for m in list_models()}


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if not np.isfinite(f) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return repr(o)


# --------------------------------------------------------------------------- analysis
def effective_dim(X: np.ndarray, var: float = 0.95, train_only_ids: np.ndarray | None = None,
                  ids: np.ndarray | None = None) -> dict:
    """PCA components needed for `var` of the variance.

    If `train_only_ids`/`ids` are given, PCA is fitted on the TRAINING videos only
    (IDs <= 1000), which is what constraint 3 requires for any fitted object.
    """
    from sklearn.decomposition import PCA

    A = np.asarray(X, dtype=np.float64)
    if A.ndim > 2:
        A = A.reshape(A.shape[0], -1)
    if ids is not None:
        mask = np.isin(ids, train_only_ids if train_only_ids is not None
                       else np.arange(1, 1001))
        A = A[mask]
    A = A - A.mean(0, keepdims=True)
    n_max = min(A.shape)
    p = PCA(n_components=n_max, svd_solver="full").fit(A)
    cum = np.cumsum(p.explained_variance_ratio_)
    k = int(np.searchsorted(cum, var) + 1)
    return {"n_fit_videos": int(A.shape[0]), "ambient_dim": int(A.shape[1]),
            f"n_components_{int(var*100)}pct": k,
            "var_explained_by_1": float(p.explained_variance_ratio_[0]),
            "var_explained_by_10": float(cum[min(9, len(cum) - 1)]),
            "participation_ratio": float(
                (p.explained_variance_.sum() ** 2) / np.sum(p.explained_variance_ ** 2))}
