"""Embedder contract. FROZEN - sub-agents must not edit this file.

An embedder is a class with:

    name      : str   unique, snake_case; becomes cache/embeddings/<name>/
    modality  : "visual" | "audio" | "audiovisual" | "handcrafted"
    source    : str   HF repo id / torchvision / open_clip tag
    revision  : str   commit hash or version pin (recorded in the manifest)
    embed(video_path) -> dict with any of:
        pooled     : (D,)      float32   one vector per video
        timeseries : (T, D)    float32   one vector per time step
        times      : (T,)      float64   seconds from onset, required with timeseries
        layers     : {name: (D,)} float32  layerwise pooled vectors

Register it so the comparison suite can find it without knowing its internals:

    from winfuse.common.embed_api import BaseEmbedder, register_embedder

    @register_embedder
    class ClipImage(BaseEmbedder):
        name = "clip_vit_b32"
        modality = "visual"
        ...

The base class validates every output: float32, finite, correct ndim, consistent D,
`len(times) == timeseries.shape[0]`, and non-empty. A violating embedder raises
`EmbedError` naming the offending key rather than writing a bad cache entry.

Device policy
-------------
MPS is permitted for any stage. `resolve_device()` returns the best available device
unless overridden by `EEG_EMBED_DEVICE`. Two safeguards remain in force, because they
are correctness gates rather than permission gates:

1. `verify_device_equivalence` embeds N videos on MPS and on CPU and requires
   per-video cosine similarity >= 0.99. Every embedder must run this before a full
   extraction, and the result goes in the manifest. On failure the caller must fall
   back to CPU and log a decision.
2. Anything that produces a REPORTED STATISTIC (ridge, RSA, CCA, CKA, permutation
   tests) must either run on CPU or pass a tighter agreement check, because float32
   GPU linear algebra can shift a p-value. See `assert_close_to_cpu`.

`PYTORCH_ENABLE_MPS_FALLBACK=1` is set by `configure_torch()`, and any op that falls
back to CPU is captured and recorded.
"""
from __future__ import annotations

import abc
import os
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np

MODALITIES = ("visual", "audio", "audiovisual", "handcrafted")

# Equivalence thresholds
DEVICE_COS_MIN = 0.99        # embeddings: per-video cosine similarity MPS vs CPU
STAT_RTOL = 1e-4             # reported statistics: relative tolerance vs CPU
STAT_ATOL = 1e-6


class EmbedError(RuntimeError):
    """Raised when an embedder violates the output contract."""


# --------------------------------------------------------------------------- device
def configure_torch() -> dict:
    """Set env flags and return a description of the torch environment."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch

        return {
            "torch": torch.__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "threads": torch.get_num_threads(),
        }
    except Exception as e:  # noqa: BLE001
        return {"torch": None, "error": repr(e)}


def resolve_device(prefer: str | None = None) -> str:
    """Best available device. `EEG_EMBED_DEVICE` overrides everything."""
    env = os.environ.get("EEG_EMBED_DEVICE")
    if env:
        return env
    if prefer:
        return prefer
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(a, b) / (na * nb))


def verify_device_equivalence(
    make_embedder,
    video_paths: Iterable[str | Path],
    key: str = "pooled",
    cos_min: float = DEVICE_COS_MIN,
) -> dict:
    """Embed the same videos on MPS and CPU; require cosine >= cos_min per video.

    `make_embedder` is a zero-arg callable returning a fresh embedder bound to the
    device named in `EEG_EMBED_DEVICE` (set and restored by this function).

    Returns a manifest-ready dict. `passed=False` means the caller MUST use CPU and
    log a decision saying so.
    """
    paths = [str(p) for p in video_paths]
    out: dict[str, Any] = {"key": key, "cos_min_required": cos_min,
                           "n_videos": len(paths)}
    try:
        import torch

        if not torch.backends.mps.is_available():
            out.update({"passed": False, "skipped": True,
                        "reason": "MPS not available on this machine"})
            return out
    except Exception as e:  # noqa: BLE001
        out.update({"passed": False, "skipped": True, "reason": f"torch import failed: {e!r}"})
        return out

    prev = os.environ.get("EEG_EMBED_DEVICE")
    vecs: dict[str, list[np.ndarray]] = {}
    timings: dict[str, float] = {}
    try:
        for dev in ("cpu", "mps"):
            os.environ["EEG_EMBED_DEVICE"] = dev
            emb = make_embedder()
            t0 = time.perf_counter()
            got = []
            for p in paths:
                r = emb.embed(p)
                v = r.get(key)
                if v is None:
                    raise EmbedError(f"embedder returned no {key!r} for {p}")
                got.append(np.asarray(v, dtype=np.float64).ravel())
            timings[dev] = time.perf_counter() - t0
            vecs[dev] = got
            del emb
    except Exception as e:  # noqa: BLE001
        out.update({"passed": False, "error": repr(e)})
        return out
    finally:
        if prev is None:
            os.environ.pop("EEG_EMBED_DEVICE", None)
        else:
            os.environ["EEG_EMBED_DEVICE"] = prev

    cs = [cosine(a, b) for a, b in zip(vecs["cpu"], vecs["mps"])]
    maxabs = [float(np.abs(a - b).max()) for a, b in zip(vecs["cpu"], vecs["mps"])]
    out.update({
        "passed": bool(min(cs) >= cos_min),
        "cosine_min": float(np.min(cs)),
        "cosine_mean": float(np.mean(cs)),
        "cosine_per_video": [round(c, 6) for c in cs],
        "max_abs_diff": float(np.max(maxabs)),
        "seconds_cpu": round(timings.get("cpu", float("nan")), 3),
        "seconds_mps": round(timings.get("mps", float("nan")), 3),
        "speedup_mps": (round(timings["cpu"] / timings["mps"], 2)
                        if timings.get("mps") else None),
    })
    return out


def assert_close_to_cpu(name: str, gpu_val, cpu_val,
                        rtol: float = STAT_RTOL, atol: float = STAT_ATOL) -> dict:
    """Gate for moving a REPORTED STATISTIC onto MPS.

    Returns a dict for the results file. `passed=False` means the statistic must be
    taken from the CPU path. Never silently prefer the GPU number.
    """
    g = np.asarray(gpu_val, dtype=np.float64)
    c = np.asarray(cpu_val, dtype=np.float64)
    if g.shape != c.shape:
        return {"stat": name, "passed": False, "reason": f"shape {g.shape} vs {c.shape}"}
    ok = bool(np.allclose(g, c, rtol=rtol, atol=atol, equal_nan=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.nanmax(np.abs(g - c) / np.maximum(np.abs(c), atol))
    return {"stat": name, "passed": ok, "rtol": rtol, "atol": atol,
            "max_abs_diff": float(np.nanmax(np.abs(g - c))),
            "max_rel_diff": float(rel)}


class MpsFallbackRecorder:
    """Context manager capturing torch warnings about ops falling back to CPU."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self._cm = None

    def __enter__(self) -> "MpsFallbackRecorder":
        self._cm = warnings.catch_warnings(record=True)
        self._caught = self._cm.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc) -> None:
        for w in self._caught:
            msg = str(w.message)
            if "fallback" in msg.lower() or "not currently supported" in msg.lower() \
                    or "aten::" in msg:
                self.messages.append(msg[:300])
        self._cm.__exit__(*exc)

    @property
    def fell_back(self) -> bool:
        return bool(self.messages)

    def summary(self) -> dict:
        return {"mps_fallback_ops": sorted(set(self.messages)),
                "n_fallback_warnings": len(self.messages)}


# --------------------------------------------------------------------------- registry
_REG: dict[str, type["BaseEmbedder"]] = {}

EMBEDDER_PACKAGES = ("winfuse.embed_visual", "winfuse.embed_audio")


def register_embedder(cls: type["BaseEmbedder"]) -> type["BaseEmbedder"]:
    nm = getattr(cls, "name", None)
    if not nm or not isinstance(nm, str):
        raise TypeError(f"{cls.__name__} must define a non-empty string `name`")
    if getattr(cls, "modality", None) not in MODALITIES:
        raise TypeError(f"{cls.__name__}.modality must be one of {MODALITIES}")
    if nm in _REG and _REG[nm] is not cls:
        raise ValueError(f"embedder name {nm!r} already registered by "
                         f"{_REG[nm].__module__}.{_REG[nm].__name__}")
    _REG[nm] = cls
    return cls


def discover_embedders(verbose: bool = False) -> dict[str, str]:
    """Import embedder packages so @register_embedder runs. Tolerates failures."""
    import importlib

    status: dict[str, str] = {}
    for pkg in EMBEDDER_PACKAGES:
        try:
            importlib.import_module(pkg)
            status[pkg] = "ok"
        except Exception as e:  # noqa: BLE001
            status[pkg] = repr(e)
        if verbose:
            print(f"[discover_embedders] {pkg}: {status[pkg]}")
    return status


def list_embedders(auto: bool = True) -> list[str]:
    if auto and not _REG:
        discover_embedders()
    return sorted(_REG)


def get_embedder(name: str, auto: bool = True, **kw: Any) -> "BaseEmbedder":
    if auto and name not in _REG:
        discover_embedders()
    if name not in _REG:
        raise KeyError(f"no embedder named {name!r}; known: {sorted(_REG)}")
    return _REG[name](**kw)


def embedder_class(name: str, auto: bool = True) -> type["BaseEmbedder"]:
    if auto and name not in _REG:
        discover_embedders()
    if name not in _REG:
        raise KeyError(f"no embedder named {name!r}; known: {sorted(_REG)}")
    return _REG[name]


# --------------------------------------------------------------------------- base
class BaseEmbedder(abc.ABC):
    """Base class for all stimulus embedders."""

    name: str = ""
    modality: str = "visual"
    description: str = ""
    source: str = ""          # HF repo id, torchvision, open_clip tag
    revision: str = ""        # commit hash / version pin
    agent: str = ""           # "V-A" or "V-B"
    frame_variant: str = "pooled"   # which src.common.frames variant it consumes
    optional: bool = False    # True for models that may not fit in 16 GB

    def __init__(self, device: str | None = None, **params: Any) -> None:
        self.params = dict(params)
        self.device = resolve_device(device)
        self._loaded = False

    # ---------------------------------------------------------------- hooks
    def _load(self) -> None:
        """Load weights. Called lazily on first embed()."""
        return None

    @abc.abstractmethod
    def _embed(self, video_path: str) -> dict:
        """Return a dict with any of pooled / timeseries+times / layers."""

    # ---------------------------------------------------------------- contract
    def embed(self, video_path: str | Path) -> dict:
        if not self._loaded:
            self._load()
            self._loaded = True
        out = self._embed(str(video_path))
        return self._check(out, video_path)

    def _check(self, out: dict, video_path) -> dict:
        if not isinstance(out, dict):
            raise EmbedError(f"{self.name}: _embed must return a dict, got {type(out)}")
        keys = {"pooled", "timeseries", "times", "layers", "meta"}
        extra = set(out) - keys
        if extra:
            raise EmbedError(f"{self.name}: unexpected keys {sorted(extra)}; allowed {sorted(keys)}")
        if not ({"pooled", "timeseries", "layers"} & set(out)):
            raise EmbedError(f"{self.name}: produced none of pooled/timeseries/layers "
                             f"for {video_path}")

        res: dict[str, Any] = {}
        if "pooled" in out and out["pooled"] is not None:
            v = np.asarray(out["pooled"], dtype=np.float32).ravel()
            self._finite(v, "pooled", video_path)
            if v.size == 0:
                raise EmbedError(f"{self.name}: empty pooled vector for {video_path}")
            res["pooled"] = np.ascontiguousarray(v)

        if "timeseries" in out and out["timeseries"] is not None:
            M = np.asarray(out["timeseries"], dtype=np.float32)
            if M.ndim != 2:
                raise EmbedError(f"{self.name}: timeseries must be (T, D), got {M.shape}")
            self._finite(M, "timeseries", video_path)
            res["timeseries"] = np.ascontiguousarray(M)
            t = out.get("times")
            if t is None:
                raise EmbedError(f"{self.name}: timeseries requires `times`")
            t = np.asarray(t, dtype=np.float64).ravel()
            if t.size != M.shape[0]:
                raise EmbedError(f"{self.name}: len(times)={t.size} != "
                                 f"timeseries rows {M.shape[0]}")
            res["times"] = t

        if "layers" in out and out["layers"] is not None:
            if not isinstance(out["layers"], dict):
                raise EmbedError(f"{self.name}: layers must be a dict")
            lay = {}
            for k, v in out["layers"].items():
                a = np.asarray(v, dtype=np.float32).ravel()
                self._finite(a, f"layers[{k}]", video_path)
                if a.size == 0:
                    raise EmbedError(f"{self.name}: empty layer {k!r} for {video_path}")
                lay[str(k)] = np.ascontiguousarray(a)
            if not lay:
                raise EmbedError(f"{self.name}: layers dict is empty")
            res["layers"] = lay

        if "meta" in out:
            res["meta"] = out["meta"]
        return res

    def _finite(self, a: np.ndarray, what: str, video_path) -> None:
        if not np.isfinite(a).all():
            n = int((~np.isfinite(a)).sum())
            raise EmbedError(f"{self.name}: {n} non-finite values in {what} for {video_path}")

    # ---------------------------------------------------------------- info
    def describe(self) -> dict:
        return {
            "name": self.name,
            "class": f"{type(self).__module__}.{type(self).__name__}",
            "modality": self.modality,
            "description": self.description,
            "source": self.source,
            "revision": self.revision,
            "agent": self.agent,
            "frame_variant": self.frame_variant,
            "optional": bool(self.optional),
            "device": self.device,
            "params": {k: (v if isinstance(v, (str, int, float, bool, type(None))) else repr(v))
                       for k, v in self.params.items()},
        }

    def n_params(self) -> int | None:
        """Parameter count, if the embedder holds a torch module in `self.model`."""
        m = getattr(self, "model", None)
        if m is None:
            return None
        try:
            return int(sum(p.numel() for p in m.parameters()))
        except Exception:
            return None
