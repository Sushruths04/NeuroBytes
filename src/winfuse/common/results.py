"""Results writing. FROZEN CONTRACT - sub-agents must not edit this file.

Writes, for each method:
    results/<method>/metrics.json   machine readable, schema in results_schema.md
    results/<method>/summary.md     human readable, generated from the same dict

Usage
-----
    from winfuse.common.results import ResultsWriter

    rw = ResultsWriter("band_power", agent="A")
    rw.set_meta(n_features=85, description="Welch PSD band powers", params={...})
    rw.add_subject("sub-01", {
        "runtime_s": {"fit": 1.2, "transform": 3.4},
        "peak_rss_mb": 512.0,
        "reliability": {"split_half_r": 0.81, "spearman_brown": 0.90},
        ...
    })
    rw.write()          # -> results/band_power/{metrics.json,summary.md}

Nothing here computes a metric. It only stores and renders what you pass in, so a
number in a report can always be traced to the code that produced it.
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import RESULTS_DIR

SCHEMA_VERSION = "1.0"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return repr(obj)


def _env() -> dict:
    def ver(mod: str) -> str | None:
        try:
            m = __import__(mod)
            return getattr(m, "__version__", None)
        except Exception:
            return None

    try:
        git = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        rev = git.stdout.strip() if git.returncode == 0 else None
    except Exception:
        rev = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_rev": rev,
        "numpy": ver("numpy"),
        "scipy": ver("scipy"),
        "sklearn": ver("sklearn"),
        "mne": ver("mne"),
        "pyriemann": ver("pyriemann"),
        "antropy": ver("antropy"),
        "pywt": ver("pywt"),
    }


class ResultsWriter:
    """Accumulates per-subject metric dicts for one method and writes both outputs."""

    def __init__(self, method: str, agent: str = "", root: Path | None = None) -> None:
        if not method or "/" in method or " " in method:
            raise ValueError(f"method must be a non-empty path-safe token, got {method!r}")
        self.method = method
        self.agent = agent
        self.root = (root or RESULTS_DIR) / method
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta: dict[str, Any] = {}
        self.subjects: dict[str, dict] = {}
        self.aggregate: dict[str, Any] = {}
        self.notes: list[str] = []
        self.figures: list[str] = []

    # ------------------------------------------------------------------ setters
    def set_meta(self, **kwargs: Any) -> "ResultsWriter":
        self.meta.update(_jsonable(kwargs))
        return self

    def add_subject(self, subject: str, metrics: dict) -> "ResultsWriter":
        self.subjects[subject] = _jsonable(metrics)
        return self

    def set_aggregate(self, **kwargs: Any) -> "ResultsWriter":
        self.aggregate.update(_jsonable(kwargs))
        return self

    def add_note(self, text: str) -> "ResultsWriter":
        """A caveat, limitation or 'this could not be computed because ...' statement."""
        self.notes.append(str(text))
        return self

    def add_figure(self, path: str | Path) -> "ResultsWriter":
        self.figures.append(str(path))
        return self

    # ------------------------------------------------------------------ output
    def payload(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "method": self.method,
            "agent": self.agent,
            "generated": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "environment": _env(),
            "meta": self.meta,
            "subjects": self.subjects,
            "aggregate": self.aggregate,
            "notes": self.notes,
            "figures": self.figures,
        }

    def write(self) -> tuple[Path, Path]:
        d = self.payload()
        jp = self.root / "metrics.json"
        jp.write_text(json.dumps(d, indent=2))
        mp = self.root / "summary.md"
        mp.write_text(render_summary(d))
        return jp, mp


# ---------------------------------------------------------------------- rendering
# Metric keys promoted into the headline per-subject table, in display order.
HEADLINE = [
    ("n_features", "n_feat", "{}"),
    ("runtime_s.transform", "transform s", "{:.2f}"),
    ("peak_rss_mb", "peak RSS MB", "{:.0f}"),
    ("blink.mean_abs_auc", "blink AUC (mean)", "{:.3f}"),
    ("blink.max_abs_auc", "blink AUC (max)", "{:.3f}"),
    ("blink.frac_trials_with_blink", "blink trials", "{:.3f}"),
    ("reliability.spearman_brown", "split-half (SB)", "{:.3f}"),
    ("erp.snr", "ERP SNR", "{:.2f}"),
    ("erp.peak_latency_s", "ERP peak s", "{:.3f}"),
    ("retrieval_proxy.top1", "proxy top-1", "{:.3f}"),
    ("retrieval_proxy.top5", "proxy top-5", "{:.3f}"),
    ("retrieval_proxy.top1_null", "proxy top-1 null", "{:.3f}"),
    ("final_dev.top1", "final top-1", "{:.3f}"),
    ("final_dev.top5", "final top-5", "{:.3f}"),
]


def dig(d: dict, dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _fmt(v: Any, spec: str) -> str:
    if v is None:
        return "–"
    try:
        return spec.format(v)
    except (ValueError, TypeError):
        return str(v)


def render_summary(d: dict) -> str:
    L: list[str] = []
    A = L.append
    meta = d.get("meta", {})
    A(f"# {d['method']}\n")
    if meta.get("description"):
        A(f"{meta['description']}\n")
    A(f"- Agent: `{d.get('agent') or 'unknown'}`")
    A(f"- Generated: `{d['generated']}`")
    if meta.get("n_features") is not None:
        A(f"- Features per trial: **{meta['n_features']}**")
    if meta.get("requires_raw"):
        A("- Consumes **raw 500 Hz** epochs (not the standard preprocessed input)")
    env = d.get("environment", {})
    A(f"- Environment: python {env.get('python')}, numpy {env.get('numpy')}, "
      f"scipy {env.get('scipy')}, sklearn {env.get('sklearn')}"
      + (f", git `{env['git_rev']}`" if env.get("git_rev") else ""))
    A("")

    subs = d.get("subjects", {})
    if subs:
        cols = [(k, lbl, spec) for k, lbl, spec in HEADLINE
                if any(dig(v, k) is not None for v in subs.values())]
        if cols:
            A("## Per-subject metrics\n")
            A("| subject | " + " | ".join(lbl for _, lbl, _ in cols) + " |")
            A("|---" * (len(cols) + 1) + "|")
            for sub in sorted(subs):
                row = [_fmt(dig(subs[sub], k), spec) for k, _, spec in cols]
                A(f"| {sub} | " + " | ".join(row) + " |")
            A("")

    if d.get("aggregate"):
        A("## Aggregate\n")
        A("| metric | value |")
        A("|---|---|")
        for k, v in d["aggregate"].items():
            A(f"| {k} | {_fmt_any(v)} |")
        A("")

    if meta.get("params"):
        A("## Parameters\n")
        A("```json")
        A(json.dumps(meta["params"], indent=2))
        A("```\n")

    # anything per-subject that did not fit the headline table
    extra_keys = sorted({k for v in subs.values() for k in v} - {c.split(".")[0] for c, _, _ in HEADLINE})
    if extra_keys:
        A("## Additional per-subject detail\n")
        for sub in sorted(subs):
            present = {k: subs[sub][k] for k in extra_keys if k in subs[sub]}
            if not present:
                continue
            A(f"### {sub}\n")
            A("```json")
            A(json.dumps(present, indent=2))
            A("```\n")

    if d.get("figures"):
        A("## Figures\n")
        for f in d["figures"]:
            A(f"- `{f}`")
        A("")

    if d.get("notes"):
        A("## Notes, caveats and what could not be computed\n")
        for n in d["notes"]:
            A(f"- {n}")
        A("")

    A("---")
    A(f"Machine-readable source: `results/{d['method']}/metrics.json` "
      f"(schema `{d.get('schema_version')}`, documented in `src/common/results_schema.md`).")
    return "\n".join(L) + "\n"


def _fmt_any(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (dict, list)):
        return f"`{json.dumps(v)}`"
    return str(v)


def load_all(root: Path | None = None) -> dict[str, dict]:
    """Read every results/*/metrics.json. Skips unreadable ones instead of failing."""
    root = root or RESULTS_DIR
    out: dict[str, dict] = {}
    if not root.exists():
        return out
    for p in sorted(root.glob("*/metrics.json")):
        try:
            out[p.parent.name] = json.loads(p.read_text())
        except Exception:
            continue
    return out
