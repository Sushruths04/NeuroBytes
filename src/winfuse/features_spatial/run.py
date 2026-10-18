"""Benchmark + run Agent B's spatial extractors.

    .venv-eeg/bin/python -m winfuse.features_spatial.run --subject sub-01
    .venv-eeg/bin/python -m winfuse.features_spatial.run --all-subjects
    .venv-eeg/bin/python -m winfuse.features_spatial.run --subject sub-01 --methods cov_tangent,cov_vec

Prints a per-method table of n_features / fit s / transform s / peak RSS MB and,
unless --no-write, writes results/<name>/{metrics.json,summary.md} via
ResultsWriter and caches each feature matrix to cache/features/.

Only development subjects (sub-01..04) are touched. Fitting is on TRAIN only.
No Agent-C validation metrics are fabricated here.
"""
from __future__ import annotations

import argparse
import gc
from dataclasses import asdict

import numpy as np

from winfuse.common import preprocess as pp
from winfuse.common.config import CACHE_DIR, DEV_SUBJECTS, SEED
from winfuse.common.feature_api import discover, get_extractor, list_extractors
from winfuse.common.preprocess import PreprocConfig
from winfuse.common.results import ResultsWriter

from ._util import peak_rss_mb, timer

ORDER = [
    "cov_tangent",
    "cov_vec",
    "log_variance",
    "cov_tangent_ea",
    "xdawn_evoked",
    "spatial_pca",
    "ica_components",
    "nonlinear_entropy",
]

_FEAT_CACHE = CACHE_DIR / "features"
_FEAT_CACHE.mkdir(parents=True, exist_ok=True)


def _agent_b_methods() -> list[str]:
    discover()
    all_reg = set(list_extractors())
    return [m for m in ORDER if m in all_reg]


def run_subject(sub: str, methods: list[str], write: bool, use_feat_cache: bool):
    np.random.seed(SEED)
    X, ids = pp.preprocessed_train(sub)
    n_trials = X.shape[0]
    rows = []
    results: dict[str, dict] = {}
    for name in methods:
        ex = get_extractor(name)
        with timer() as tf:
            ex.fit(X)
        with timer() as tt:
            F = ex.transform(X)
        rss = peak_rss_mb()
        nfeat = F.shape[1]
        if use_feat_cache:
            np.save(_FEAT_CACHE / f"{name}_{sub}.npy", F)
        rows.append((name, nfeat, tf[0], tt[0], rss))
        results[name] = {
            "n_trials": int(n_trials),
            "n_features": int(nfeat),
            "runtime_s": {"fit": round(tf[0], 4), "transform": round(tt[0], 4),
                          "total": round(tf[0] + tt[0], 4)},
            "peak_rss_mb": round(rss, 1),
            "describe": ex.describe(),
            "feature_names_head": ex.feature_names()[:20],
        }
        del F
        gc.collect()
    del X
    gc.collect()
    return rows, results


def main() -> None:
    ap = argparse.ArgumentParser(description="Agent B spatial feature benchmark")
    ap.add_argument("--subject", default="sub-01")
    ap.add_argument("--all-subjects", action="store_true",
                    help="run every dev subject (sub-01..04)")
    ap.add_argument("--methods", default="",
                    help="comma-separated extractor names; default = all Agent B methods")
    ap.add_argument("--no-write", action="store_true", help="skip writing results/")
    ap.add_argument("--no-feature-cache", action="store_true",
                    help="skip caching feature matrices to cache/features/")
    args = ap.parse_args()

    methods = args.methods.split(",") if args.methods else _agent_b_methods()
    methods = [m.strip() for m in methods if m.strip()]
    subjects = list(DEV_SUBJECTS) if args.all_subjects else [args.subject]

    write = not args.no_write
    use_feat_cache = not args.no_feature_cache

    # Accumulate per-method results across subjects for a single ResultsWriter each.
    per_method: dict[str, dict[str, dict]] = {m: {} for m in methods}
    cfg = PreprocConfig()

    for sub in subjects:
        print(f"\n=== {sub} ===")
        rows, results = run_subject(sub, methods, write, use_feat_cache)
        hdr = f"{'method':20s} {'n_feat':>7s} {'fit s':>9s} {'xform s':>9s} {'peak RSS MB':>12s}"
        print(hdr)
        print("-" * len(hdr))
        for name, nfeat, fs, ts, rss in rows:
            print(f"{name:20s} {nfeat:7d} {fs:9.3f} {ts:9.3f} {rss:12.1f}")
        for name, res in results.items():
            per_method[name][sub] = res

    if write:
        for name in methods:
            ex = get_extractor(name)
            desc = ex.describe()
            rw = ResultsWriter(name, agent="B")
            rw.set_meta(
                description=desc["description"],
                n_features=per_method[name][subjects[0]]["n_features"],
                params=desc["params"],
                requires_raw=desc["requires_raw"],
                preprocessing=asdict(cfg),
                feature_names_head=ex.feature_names()[:20],
            )
            for sub in subjects:
                r = per_method[name][sub]
                rw.add_subject(sub, {
                    "n_trials": r["n_trials"],
                    "n_features": r["n_features"],
                    "runtime_s": r["runtime_s"],
                    "peak_rss_mb": r["peak_rss_mb"],
                })
            rw.add_note(
                "Agent B produced only cost/shape metrics here. Blink/ERP/"
                "reliability/retrieval validation is Agent C's responsibility and is "
                "intentionally omitted (not fabricated)."
            )
            rw.add_note(f"Fitted on TRAIN only; subjects run: {', '.join(subjects)}.")
            jp, mp = rw.write()
            print(f"wrote {jp}")


if __name__ == "__main__":
    main()
