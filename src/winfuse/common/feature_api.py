"""Feature extractor contract. FROZEN - sub-agents must not edit this file.

Every extractor is a class that subclasses `BaseExtractor` and implements:

    name          : str          unique, snake_case, stable (it becomes a directory name)
    fit(X_train)  : -> self      X_train is PREPROCESSED training epochs (n, 17, n_samples)
    transform(X)  : -> (n_trials, n_features) float32
    feature_names(): -> list[str], length == n_features

Register it so Phase 2 can find it without knowing your internals:

    from winfuse.common.feature_api import BaseExtractor, register

    @register
    class MyThing(BaseExtractor):
        name = "my_thing"
        def _fit(self, X): ...            # optional
        def _transform(self, X): ...      # required, returns (n, n_feat)
        def feature_names(self): ...      # required

Rules the base class enforces for you
-------------------------------------
* `transform` before `fit` raises.
* output is coerced to float32, C-contiguous, 2-D with the right number of rows
* output is checked for NaN/Inf, with the offending feature names reported
* `len(feature_names()) == n_features` is checked
* `fit` is only ever handed training data; there is nothing in this class that
  would let you see test data

Input convention
----------------
`fit`/`transform` receive the output of `src.common.preprocess`, i.e. band-passed,
50 Hz-notched, 250 Hz, per-channel z-scored epochs of shape (n_trials, 17, 750),
float32. If your extractor needs the raw 500 Hz signal, say so in `requires_raw`
and the harness will hand you unfiltered epochs instead.
"""
from __future__ import annotations

import abc
import time
from typing import Any, Callable, Iterable

import numpy as np

from .config import N_CHANNELS

# --------------------------------------------------------------------------- registry
_REGISTRY: dict[str, type["BaseExtractor"]] = {}


def register(cls: type["BaseExtractor"]) -> type["BaseExtractor"]:
    """Class decorator. Makes the extractor discoverable by name."""
    nm = getattr(cls, "name", None)
    if not nm or not isinstance(nm, str):
        raise TypeError(f"{cls.__name__} must define a non-empty string class attr `name`")
    if nm in _REGISTRY and _REGISTRY[nm] is not cls:
        raise ValueError(
            f"extractor name {nm!r} already registered by {_REGISTRY[nm].__module__}."
            f"{_REGISTRY[nm].__name__}; names must be unique across agents"
        )
    _REGISTRY[nm] = cls
    return cls


# Packages that may contain registered extractors. Import failures are tolerated so
# that one agent's broken module cannot block the others.
EXTRACTOR_PACKAGES = (
    "winfuse.features_time_freq",
    "winfuse.features_spatial",
)


def discover(verbose: bool = False) -> dict[str, str]:
    """Import the extractor packages so their @register decorators run.

    Returns a dict of package -> status ('ok' or the error repr).
    """
    import importlib

    status: dict[str, str] = {}
    for pkg in EXTRACTOR_PACKAGES:
        try:
            importlib.import_module(pkg)
            status[pkg] = "ok"
        except Exception as e:  # noqa: BLE001 - deliberately tolerant
            status[pkg] = repr(e)
        if verbose:
            print(f"[discover] {pkg}: {status[pkg]}")
    return status


def list_extractors(auto_discover: bool = True) -> list[str]:
    if auto_discover and not _REGISTRY:
        discover()
    return sorted(_REGISTRY)


def get_extractor(name: str, auto_discover: bool = True, **kwargs: Any) -> "BaseExtractor":
    if auto_discover and name not in _REGISTRY:
        discover()
    if name not in _REGISTRY:
        raise KeyError(f"no extractor named {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def extractor_class(name: str, auto_discover: bool = True) -> type["BaseExtractor"]:
    if auto_discover and name not in _REGISTRY:
        discover()
    if name not in _REGISTRY:
        raise KeyError(f"no extractor named {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


# --------------------------------------------------------------------------- base class
class FeatureError(RuntimeError):
    """Raised when an extractor violates the output contract."""


class BaseExtractor(abc.ABC):
    """Base class for all feature extractors.

    Subclasses override `_fit` (optional) and `_transform` (required).
    Do not override `fit`/`transform`; they implement the contract checks.
    """

    #: unique, stable, snake_case. Becomes results/<name>/ and figures/<name>/.
    name: str = ""
    #: one-line human description, used in reports
    description: str = ""
    #: which agent owns it: "A", "B" or "C"
    agent: str = ""
    #: set True if you need unfiltered 500 Hz epochs rather than the preprocessed ones
    requires_raw: bool = False
    #: free-form, written verbatim into metrics.json for provenance
    params: dict[str, Any]

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self._fitted = False
        self._n_features: int | None = None
        self._fit_seconds: float | None = None
        self._n_fit_trials: int | None = None

    # ---------------------------------------------------------------- hooks
    def _fit(self, X: np.ndarray) -> None:
        """Override if the extractor learns anything. X is TRAINING epochs only."""
        return None

    @abc.abstractmethod
    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Return (n_trials, n_features). Will be coerced to float32 by the caller."""

    @abc.abstractmethod
    def feature_names(self) -> list[str]:
        """Return exactly n_features names, in the same order as the columns."""

    # ---------------------------------------------------------------- contract
    def fit(self, X_train: np.ndarray) -> "BaseExtractor":
        X_train = self._check_input(X_train)
        t0 = time.perf_counter()
        self._fit(X_train)
        self._fit_seconds = time.perf_counter() - t0
        self._n_fit_trials = int(X_train.shape[0])
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise FeatureError(
                f"{self.name}: transform() called before fit(). Fit on training epochs first."
            )
        X = self._check_input(X)
        F = self._transform(X)
        return self._check_output(F, n_trials=X.shape[0])

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    # ---------------------------------------------------------------- helpers
    @property
    def n_features(self) -> int:
        if self._n_features is None:
            try:
                return len(self.feature_names())
            except Exception:
                raise FeatureError(f"{self.name}: n_features unknown before transform()")
        return self._n_features

    @property
    def fitted(self) -> bool:
        return self._fitted

    def describe(self) -> dict:
        return {
            "name": self.name,
            "class": f"{type(self).__module__}.{type(self).__name__}",
            "agent": self.agent,
            "description": self.description,
            "requires_raw": bool(self.requires_raw),
            "params": _jsonable(self.params),
            "fitted": self._fitted,
            "n_fit_trials": self._n_fit_trials,
            "fit_seconds": None if self._fit_seconds is None else round(self._fit_seconds, 4),
            "n_features": self._n_features,
        }

    @staticmethod
    def _check_input(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 3:
            raise FeatureError(f"expected (n_trials, n_channels, n_samples), got {X.shape}")
        if X.shape[1] != N_CHANNELS:
            raise FeatureError(f"expected {N_CHANNELS} channels, got {X.shape[1]}")
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        return np.ascontiguousarray(X)

    def _check_output(self, F: np.ndarray, n_trials: int) -> np.ndarray:
        F = np.asarray(F)
        if F.ndim == 1:
            F = F[:, None]
        if F.ndim != 2:
            raise FeatureError(f"{self.name}: _transform returned ndim={F.ndim}, expected 2")
        if F.shape[0] != n_trials:
            raise FeatureError(
                f"{self.name}: _transform returned {F.shape[0]} rows for {n_trials} trials"
            )
        F = np.ascontiguousarray(F.astype(np.float32, copy=False))

        names = list(self.feature_names())
        if len(names) != F.shape[1]:
            raise FeatureError(
                f"{self.name}: feature_names() has {len(names)} entries but _transform "
                f"produced {F.shape[1]} columns"
            )
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})[:10]
            raise FeatureError(f"{self.name}: duplicate feature names, e.g. {dupes}")

        bad = ~np.isfinite(F)
        if bad.any():
            cols = np.unique(np.where(bad)[1])[:10]
            raise FeatureError(
                f"{self.name}: {int(bad.sum())} non-finite values in output; first bad "
                f"features: {[names[c] for c in cols]}"
            )
        self._n_features = int(F.shape[1])
        return F


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion so `params` can always be serialised."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


# --------------------------------------------------------------------------- utilities
def channel_feature_names(prefix: str, n_channels: int = N_CHANNELS) -> list[str]:
    """`ch00_<prefix>` ... `ch16_<prefix>`. Channels are anonymous indices."""
    return [f"ch{c:02d}_{prefix}" for c in range(n_channels)]


def cross_names(prefix: str, inner: Iterable[str], n_channels: int = N_CHANNELS) -> list[str]:
    """Channel-major cross product: ch00_<inner0>, ch00_<inner1>, ... ch16_<innerN>."""
    inner = list(inner)
    return [f"ch{c:02d}_{prefix}_{i}" for c in range(n_channels) for i in inner]


class LambdaExtractor(BaseExtractor):
    """Tiny adapter for tests and smoke runs. Not registered."""

    name = "lambda"

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], names: list[str], **kw: Any) -> None:
        super().__init__(**kw)
        self._fn = fn
        self._names = list(names)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return self._fn(X)

    def feature_names(self) -> list[str]:
        return list(self._names)
