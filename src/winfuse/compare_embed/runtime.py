"""Runtime policy for the comparison suite: job lock, thread policy, stage timing.

Why this module exists
----------------------
A profiling run of one cell took 13m34s wall with **37m user and 64m sys** CPU time.
Sys time exceeding user time by that margin is not computation, it is contention:
BLAS was spawning 8 threads per call while Python-level loops ran concurrently, so
the machine spent more time scheduling threads than doing arithmetic. The fix is to
parallelise at exactly ONE level.

Policy (per the runtime brief)
------------------------------
1. One suite job at a time, enforced by `logs/suite.lock`. A second launch refuses
   to start and prints which PID holds the lock and since when.
2. Parallelism at one level only, selected by `--parallel`:
     "blas"   : BLAS threads = N (default 4), Python loops sequential, joblib unused.
     "joblib" : joblib workers = N over independent tasks, BLAS threads pinned to 1.
   Never both. `configure_threads` must run BEFORE numpy/sklearn are imported to
   take effect, so the CLI entry points call it first.
3. Stage timing is always on: one line per stage with elapsed seconds, and a
   machine-readable table written into the results file.
"""
from __future__ import annotations

import atexit
import datetime as _dt
import json
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "logs" / "suite.lock"

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

PARALLEL_MODES = ("blas", "joblib")


# --------------------------------------------------------------------------- threads
def configure_threads(mode: str = "blas", n: int = 4) -> dict:
    """Set the thread policy. MUST be called before numpy/sklearn import.

    Returns a dict describing what was set, for the results manifest.
    """
    if mode not in PARALLEL_MODES:
        raise ValueError(f"parallel mode must be one of {PARALLEL_MODES}, got {mode!r}")
    blas = int(n) if mode == "blas" else 1
    workers = 1 if mode == "blas" else int(n)
    for v in _THREAD_ENV_VARS:
        os.environ[v] = str(blas)
    os.environ["EEG_SUITE_WORKERS"] = str(workers)
    os.environ["EEG_SUITE_PARALLEL_MODE"] = mode
    already = "numpy" in sys.modules
    return {
        "parallel_mode": mode,
        "blas_threads": blas,
        "joblib_workers": workers,
        "env": {v: os.environ[v] for v in _THREAD_ENV_VARS},
        "numpy_already_imported": already,
        "warning": ("numpy was already imported; BLAS thread count may not have taken "
                    "effect. Call configure_threads() before importing numpy.")
        if already else None,
    }


def thread_policy() -> dict:
    """What the current process is actually using."""
    out = {v: os.environ.get(v) for v in _THREAD_ENV_VARS}
    out["parallel_mode"] = os.environ.get("EEG_SUITE_PARALLEL_MODE", "unset")
    out["joblib_workers"] = int(os.environ.get("EEG_SUITE_WORKERS", "1"))
    try:
        from threadpoolctl import threadpool_info

        out["threadpool_info"] = [
            {"prefix": d.get("prefix"), "num_threads": d.get("num_threads")}
            for d in threadpool_info()
        ]
    except Exception:
        out["threadpool_info"] = "threadpoolctl unavailable"
    return out


def n_workers() -> int:
    return int(os.environ.get("EEG_SUITE_WORKERS", "1"))


# --------------------------------------------------------------------------- lock
class LockHeld(RuntimeError):
    """Raised when another suite job already holds the lock."""


def _read_lock() -> dict | None:
    if not LOCK_PATH.exists():
        return None
    try:
        return json.loads(LOCK_PATH.read_text())
    except Exception:
        return {"raw": LOCK_PATH.read_text()[:300]}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def acquire_lock(what: str = "suite", steal_stale: bool = True) -> dict:
    """Take the single-job lock. Raises LockHeld if another live job has it.

    A lock whose PID is no longer alive is treated as stale and reclaimed, so a
    crashed job does not block the machine forever.
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    cur = _read_lock()
    if cur:
        pid = cur.get("pid")
        alive = isinstance(pid, int) and _pid_alive(pid)
        if alive:
            raise LockHeld(
                f"another suite job holds {LOCK_PATH}:\n"
                f"  pid     : {pid}\n"
                f"  job     : {cur.get('what')}\n"
                f"  argv    : {cur.get('argv')}\n"
                f"  started : {cur.get('started')}\n"
                f"Wait for it, or if you are certain it is dead remove the file."
            )
        if not steal_stale:
            raise LockHeld(f"{LOCK_PATH} exists (pid {pid} not alive); refusing to steal")
        print(f"[lock] reclaiming stale lock from dead pid {pid}", flush=True)

    rec = {"pid": os.getpid(), "what": what, "argv": sys.argv,
           "started": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "host": os.uname().nodename}
    LOCK_PATH.write_text(json.dumps(rec, indent=2))
    atexit.register(release_lock)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            prev = signal.getsignal(sig)

            def _handler(s, f, _prev=prev):
                release_lock()
                if callable(_prev):
                    _prev(s, f)
                else:
                    raise SystemExit(130)

            signal.signal(sig, _handler)
        except Exception:
            pass
    return rec


def release_lock() -> None:
    cur = _read_lock()
    if cur and cur.get("pid") == os.getpid():
        try:
            LOCK_PATH.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def job_lock(what: str = "suite"):
    acquire_lock(what)
    try:
        yield
    finally:
        release_lock()


# --------------------------------------------------------------------------- timing
class StageTimer:
    """Per-stage wall-clock timing, printed live and collected for the report."""

    def __init__(self, label: str = "", verbose: bool = True) -> None:
        self.label = label
        self.verbose = verbose
        self.stages: dict[str, float] = {}
        self.order: list[str] = []
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        if self.verbose:
            print(f"    [{self.label}] {name} ...", end="", flush=True)
        try:
            yield
        finally:
            dt = time.perf_counter() - t
            self.stages[name] = self.stages.get(name, 0.0) + dt
            if name not in self.order:
                self.order.append(name)
            if self.verbose:
                print(f" {dt:7.2f}s", flush=True)

    def add(self, name: str, seconds: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + float(seconds)
        if name not in self.order:
            self.order.append(name)

    @property
    def total(self) -> float:
        return time.perf_counter() - self._t0

    def dominant(self) -> tuple[str, float]:
        if not self.stages:
            return ("none", 0.0)
        k = max(self.stages, key=lambda s: self.stages[s])
        return (k, self.stages[k])

    def table(self) -> dict:
        tot = sum(self.stages.values()) or 1.0
        dom, dsec = self.dominant()
        return {
            "stages_s": {k: round(self.stages[k], 3) for k in self.order},
            "stage_pct": {k: round(100 * self.stages[k] / tot, 1) for k in self.order},
            "sum_stages_s": round(tot, 2),
            "wall_total_s": round(self.total, 2),
            "dominant_stage": dom,
            "dominant_stage_s": round(dsec, 2),
            "dominant_stage_pct": round(100 * dsec / tot, 1),
        }

    def render_md(self) -> str:
        t = self.table()
        L = ["| stage | seconds | % of measured |", "|---|---|---|"]
        for k in self.order:
            L.append(f"| {k} | {t['stages_s'][k]:.2f} | {t['stage_pct'][k]:.1f}% |")
        L.append(f"| **total (sum of stages)** | **{t['sum_stages_s']:.2f}** | 100% |")
        L.append("")
        L.append(f"Dominant stage: **{t['dominant_stage']}** "
                 f"({t['dominant_stage_s']:.2f}s, {t['dominant_stage_pct']:.1f}%).")
        return "\n".join(L)

    def print_summary(self) -> None:
        t = self.table()
        print(f"  [{self.label}] stage timing (total {t['sum_stages_s']:.1f}s, "
              f"wall {t['wall_total_s']:.1f}s):", flush=True)
        for k in self.order:
            print(f"      {k:22} {t['stages_s'][k]:8.2f}s  {t['stage_pct'][k]:5.1f}%")
        print(f"      -> dominant: {t['dominant_stage']} "
              f"({t['dominant_stage_pct']:.1f}%)", flush=True)
