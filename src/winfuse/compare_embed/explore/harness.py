"""Shared harness for the three improvement experiments. VALIDATION ONLY.

Every experiment must call `evaluate(name, score_fn)` so the three results are directly
comparable: same subjects, same train/val split, same bijective pools, same metric.

Rules this harness enforces
---------------------------
* **No test data.** Nothing here opens `tst/` or `tst_stim/`. Experiments are judged on
  held-out TRAINING videos only. The test split is read once, by the orchestrator, for
  whichever experiment wins.
* **Bijective pools.** 102 validation trials with 102 distinct videos, candidate set
  exactly those videos — the same one-to-one structure as a real test block, which the
  organisers confirmed may be exploited. Metrics are therefore on the same scale as the
  reported test numbers.
* **Subject independence.** `score_fn` receives one subject at a time and must not
  receive or use anything fitted on another subject's *test* data. Fitting per subject
  on that subject's own TRAINING videos is allowed and expected — that is what the
  ridges already do. Anything learned across subjects must come only from training
  videos, and the resulting procedure must apply unchanged to a subject never seen.

Baseline for reference, measured with this harness: the frozen pipeline scores
val top-5 ~40.6% with the bijection step on bijective pools.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from winfuse.common.config import DEV_SUBJECTS, RESULTS_DIR
from winfuse.compare_embed import pipeline as P

OUT = RESULTS_DIR / "compare_embed" / "explore"
OUT.mkdir(parents=True, exist_ok=True)
N_POOLS = 30
POOL_SEED = 1234


def frozen() -> dict:
    return json.loads(
        (RESULTS_DIR / "compare_embed" / "_frozen_pipeline.json").read_text())


def sinkhorn(S: np.ndarray, iters: int = 3, tau: float = 0.5) -> np.ndarray:
    """The frozen bijection step. Applied to every experiment equally."""
    S = np.asarray(S, dtype=np.float64)
    Z = (S - S.mean(1, keepdims=True)) / np.maximum(S.std(1, keepdims=True), 1e-12)
    K = np.exp(np.clip(Z / tau, -700, 700))
    for _ in range(iters):
        K = K / np.maximum(K.sum(1, keepdims=True), 1e-300)
        K = K / np.maximum(K.sum(0, keepdims=True), 1e-300)
    return K


def _ranks_square(S: np.ndarray) -> np.ndarray:
    """Rank of the truth when row i's correct candidate is column i."""
    return np.array([int(np.sum(S[i] > S[i, i])) + 1 for i in range(S.shape[0])])


def bijective_pools(n: int, repeats: int = N_POOLS, seed: int = POOL_SEED) -> list:
    rng = np.random.default_rng(seed)
    k = min(102, n)
    return [rng.choice(n, k, replace=False) for _ in range(repeats)]


def score_to_metrics(M: np.ndarray, pools: list, apply_bijection: bool = True) -> dict:
    """M is (n_val, n_val) with the truth on the diagonal."""
    t1, t5, mrr = [], [], []
    for ix in pools:
        B = M[np.ix_(ix, ix)]
        if apply_bijection:
            B = sinkhorn(B)
        r = _ranks_square(B)
        t1.append(np.mean(r <= 1)); t5.append(np.mean(r <= 5))
        mrr.append(np.mean(1.0 / r))
    return {"top1": float(np.mean(t1)), "top5": float(np.mean(t5)),
            "mrr": float(np.mean(mrr))}


def evaluate(name: str, score_fn, subjects=None, note: str = "") -> dict:
    """Run one experiment.

    `score_fn(subject) -> (M_val, n_val)` must return a square validation score matrix
    whose row i corresponds to validation trial i and whose column j corresponds to the
    video of validation trial j, so the truth lies on the diagonal. Higher is better.

    Fit on that subject's TRAINING portion only. `pipeline.split_ids` gives the split.
    """
    subjects = list(subjects or DEV_SUBJECTS)
    t0 = time.perf_counter()
    per, raw = {}, {}
    for sub in subjects:
        M = score_fn(sub)
        if isinstance(M, tuple):
            M = M[0]
        M = np.asarray(M, dtype=np.float64)
        assert M.ndim == 2 and M.shape[0] == M.shape[1], f"{name}/{sub}: {M.shape}"
        pools = bijective_pools(M.shape[0])
        per[sub] = score_to_metrics(M, pools, True)
        raw[sub] = score_to_metrics(M, pools, False)
        print(f"    {sub}: val top-5 {per[sub]['top5']*100:5.2f}% "
              f"(without bijection {raw[sub]['top5']*100:5.2f}%)", flush=True)
    agg = {k: float(np.mean([per[s][k] for s in subjects])) for k in ("top1", "top5", "mrr")}
    agg_raw = {k: float(np.mean([raw[s][k] for s in subjects])) for k in ("top1", "top5")}
    res = {"name": name, "note": note, "subjects": subjects,
           "pooled_with_bijection": agg, "pooled_without_bijection": agg_raw,
           "per_subject": per, "seconds": round(time.perf_counter() - t0, 1),
           "protocol": ("validation only; bijective pools of 102 built from held-out "
                        "TRAINING videos; no test data touched")}
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"  [{name}] val top-5 {agg['top5']*100:.2f}%  top-1 {agg['top1']*100:.2f}%  "
          f"({res['seconds']:.0f}s) -> {p.name}", flush=True)
    return res


def frozen_baseline(subjects=None) -> dict:
    """The current frozen pipeline, scored through this exact harness."""
    fz = frozen()
    g = fz["channels"]
    w = np.asarray(fz["weights"], dtype=np.float64)
    tf = P.TRANSFORMS[fz["transform"]]
    cache = {}

    def fn(sub):
        if sub not in cache:
            per = P.load(sub, g)
            cache[sub] = sum(w[i] * tf(per[s]["val"]) for i, s in enumerate(g))
        return cache[sub]

    return evaluate("00_frozen_baseline", fn, subjects,
                    note="the current frozen pipeline, for reference")


if __name__ == "__main__":
    frozen_baseline()
