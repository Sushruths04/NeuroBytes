"""CLI: extract + cache Agent A time/frequency features and benchmark them.

Usage
-----
    .venv-eeg/bin/python -m winfuse.features_time_freq.run --subject sub-01
    .venv-eeg/bin/python -m winfuse.features_time_freq.run --all-subjects
    .venv-eeg/bin/python -m winfuse.features_time_freq.run --subject sub-02 --methods band_power,hjorth

For each (method, subject) it:
  * fits on the subject's TRAIN epochs, transforms them,
  * caches features to cache/features/<method>/<sub>.npy (+ names.json),
  * records n_features / fit s / transform s / peak RSS MB / cache size,
and prints a per-method table. It also writes results/<method>/metrics.json +
summary.md via the frozen ResultsWriter (runtime + provenance only; validation
metrics are left to Agent C).

Peak RSS is read from resource.getrusage(RUSAGE_SELF).ru_maxrss. On macOS this is
in BYTES; on Linux it is in kibibytes. The platform is detected and normalised to
MB. The value is a high-water mark for the whole process, so it is reported as a
monotonic peak after each extractor rather than a per-call delta.
"""
from __future__ import annotations

import argparse
import gc
import resource
import sys
import time

import numpy as np

from winfuse.common import preprocess as pp
from winfuse.common.config import ALL_SUBJECTS, DEV_SUBJECTS, HELDOUT_SUBJECTS
from winfuse.common.feature_api import get_extractor, list_extractors
from winfuse.common.preprocess import PreprocConfig
from winfuse.common.results import ResultsWriter

from . import cache as fcache

# Agent A's extractors, in a sensible display order.
AGENT_A_METHODS = [
    "raw_signal",
    "erp_windows",
    "hjorth",
    "time_stats",
    "band_power",
    "rel_band_power",
    "tf_band_power",
    "tf_band_power_reduced",
    "ersp_trialmean",
]


def _peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KiB.
    if sys.platform == "darwin":
        return ru / (1024.0 * 1024.0)
    return ru / 1024.0


def run_subject(subject: str, methods: list[str], write_results: bool = True) -> dict:
    """Run the given methods for one subject. Returns a dict of per-method rows."""
    X, ids = pp.preprocessed_train(subject)
    n_trials = X.shape[0]
    rows: dict[str, dict] = {}

    for method in methods:
        ex = get_extractor(method)
        gc.collect()
        t0 = time.perf_counter()
        ex.fit(X)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        F = ex.transform(X)
        transform_s = time.perf_counter() - t0

        names = ex.feature_names()
        path = fcache.save_features(method, subject, F, names)
        size_mb = fcache.cache_size_bytes(method, subject) / (1024.0 * 1024.0)
        peak_mb = _peak_rss_mb()

        rows[method] = {
            "n_trials": int(n_trials),
            "n_features": int(F.shape[1]),
            "fit_s": fit_s,
            "transform_s": transform_s,
            "peak_rss_mb": peak_mb,
            "cache_mb": size_mb,
            "cache_path": str(path),
            "feature_names_head": names[:20],
            "description": ex.description,
            "params": ex.params,
            "requires_raw": bool(ex.requires_raw),
        }
        del F
        gc.collect()

    if write_results:
        _write_results(subject, rows)
    del X
    gc.collect()
    return rows


def _write_results(subject: str, rows: dict) -> None:
    """Merge this subject's rows into each method's results/<method>/metrics.json."""
    import json
    from pathlib import Path

    from winfuse.common.config import RESULTS_DIR

    prep_cfg = PreprocConfig().__dict__ if hasattr(PreprocConfig(), "__dict__") else {}
    # dataclass -> dict
    from dataclasses import asdict
    prep_cfg = asdict(PreprocConfig())

    for method, row in rows.items():
        # Load existing payload if present so multiple subject runs accumulate.
        existing_subjects = {}
        jp = RESULTS_DIR / method / "metrics.json"
        if jp.exists():
            try:
                existing_subjects = json.loads(jp.read_text()).get("subjects", {})
            except Exception:
                existing_subjects = {}

        rw = ResultsWriter(method, agent="A")
        rw.set_meta(
            description=row["description"],
            n_features=row["n_features"],
            params=row["params"],
            requires_raw=row["requires_raw"],
            preprocessing=prep_cfg,
            feature_names_head=row["feature_names_head"],
        )
        # restore prior subjects, then add/overwrite this one
        for sub, m in existing_subjects.items():
            rw.add_subject(sub, m)
        rw.add_subject(subject, {
            "n_trials": row["n_trials"],
            "n_features": row["n_features"],
            "runtime_s": {
                "fit": round(row["fit_s"], 4),
                "transform": round(row["transform_s"], 4),
                "total": round(row["fit_s"] + row["transform_s"], 4),
            },
            "peak_rss_mb": round(row["peak_rss_mb"], 1),
            "cache_mb": round(row["cache_mb"], 3),
        })
        rw.add_note(
            "Validation metrics (blink/erp/reliability/retrieval/final_dev) are "
            "intentionally omitted; Agent C computes them. Runtime and provenance only."
        )
        rw.add_note(
            "peak_rss_mb is a process high-water mark (resource.ru_maxrss), not a "
            "per-extractor delta; extractors run in the same process so later rows "
            "inherit the peak of earlier ones."
        )
        rw.write()


def _print_table(subject: str, rows: dict) -> None:
    print(f"\n=== {subject} ===")
    hdr = f"{'method':24s} {'n_feat':>7s} {'fit s':>8s} {'trans s':>8s} {'peakRSS MB':>11s} {'cache MB':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for method, r in rows.items():
        print(f"{method:24s} {r['n_features']:7d} {r['fit_s']:8.2f} {r['transform_s']:8.2f} "
              f"{r['peak_rss_mb']:11.1f} {r['cache_mb']:9.3f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Agent A time/frequency feature runner")
    ap.add_argument("--subject", default="sub-01",
                    help="subject id, e.g. sub-01 (ignored if --all-subjects)")
    ap.add_argument("--all-subjects", action="store_true",
                    help="run all six subjects (train only; sub-05/06 never touch test)")
    ap.add_argument("--methods", default="",
                    help="comma-separated method names; default = all Agent A methods")
    ap.add_argument("--no-write", action="store_true",
                    help="do not write results/<method>/ json+md (benchmark only)")
    args = ap.parse_args(argv)

    if args.methods.strip():
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        known = set(list_extractors())
        unknown = [m for m in methods if m not in known]
        if unknown:
            ap.error(f"unknown methods: {unknown}; known: {sorted(known)}")
    else:
        methods = AGENT_A_METHODS

    if args.all_subjects:
        subjects = list(ALL_SUBJECTS)
    else:
        subjects = [args.subject]

    for sub in subjects:
        if sub in HELDOUT_SUBJECTS:
            # Allowed: we only ever load TRAIN epochs. io.py refuses their test files.
            print(f"[note] {sub} is held-out; using TRAIN epochs only (no test access).")
        rows = run_subject(sub, methods, write_results=not args.no_write)
        _print_table(sub, rows)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
