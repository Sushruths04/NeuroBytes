"""Independent end-to-end check on the real test recordings, and the submission path.

    .venv-embed/bin/python -m winfuse.compare_embed.verify_on_test --subjects sub-01,sub-02,sub-03,sub-04

What this does differently from `pipeline --test`
-------------------------------------------------
`pipeline --test` fuses score matrices that were computed during `--build` and read
back from `cache/pipeline/`. This module ignores that cache completely. For each
subject it:

  1. loads the raw training recordings and that subject's raw test recordings
     (`tst/sub-XX_tst.npy`) straight off disk,
  2. refits all 509 ridge scorers on all of that subject's training videos,
  3. applies them to the 102 test recordings,
  4. fuses with the frozen weight vector,
  5. writes a real submission file -- `sub-XX_ranked_candidate_list.npy`, int,
     shape (102, 102), stimulus IDs 1001..1102, rows in the recordings' original
     file order,
  6. validates that file's format, then scores it against
     `tst_stim/sub-XX_tst_stim.npy`.

So it exercises the actual submission-generation path and checks the published number
against a from-scratch recomputation. If the caching layer or the row ordering were
wrong, this would disagree.

It only runs for subjects whose labels exist (sub-01..04). sub-05/06 remain blocked by
the leakage guard.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from winfuse.common import io as cio
from winfuse.common import preprocess as pp
from winfuse.common.config import DEV_SUBJECTS, RESULTS_DIR, ROOT
from winfuse.compare_embed import fusion as fu
from winfuse.compare_embed import pipeline as P

OUTDIR = ROOT / "results" / "verify_on_test"
TEST_IDS = list(range(1001, 1103))


def validate_submission(a: np.ndarray, n_rows: int = 102) -> None:
    """The format the challenge asks for. Raises on anything wrong."""
    if a.shape != (n_rows, 102):
        raise ValueError(f"shape {a.shape}, expected ({n_rows}, 102)")
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"dtype {a.dtype} is not integer")
    want = set(TEST_IDS)
    for i in range(a.shape[0]):
        row = a[i]
        if set(row.tolist()) != want:
            raise ValueError(f"row {i} is not a permutation of 1001..1102")
    if np.unique(a, axis=0).shape[0] < 2:
        raise ValueError("every row is identical, which cannot be a real ranking")


def build_and_rank(sub: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ranked_ids (102,102) int, truth_ids (102,)) for one subject."""
    fz = json.loads((RESULTS_DIR / "compare_embed" / "_frozen_pipeline.json").read_text())
    specs, weights = fz["channels"], np.asarray(fz["weights"], dtype=np.float64)
    tf = P.TRANSFORMS[fz["transform"]]

    eeg_sets = sorted({s.split(":")[0] for s in specs})
    stims = sorted({s.split(":")[1] for s in specs})

    # ---- training side, read fresh
    F_tr, ids0 = P.eeg_matrix(sub, eeg_sets[0], "train")
    stim_tr, common = {}, None
    for st in stims:
        X, found = fu.stimulus_matrix(st, ids0, "train")
        stim_tr[st] = (X, found)
        common = set(found) if common is None else (common & set(found))
    common = sorted(common)

    def rows(member_ids, M, want):
        pos = {v: i for i, v in enumerate(member_ids)}
        return M[[pos[v] for v in want]]

    Ftr = {}
    for e in eeg_sets:
        X, mi = P.eeg_matrix(sub, e, "train")
        Ftr[e] = rows(mi, X, common)
    Etr = {st: rows(v[1], v[0], common) for st, v in stim_tr.items()}

    # ---- test side, read fresh from tst/sub-XX_tst.npy
    Fte = {e: P.eeg_matrix(sub, e, "test") for e in eeg_sets}
    Ecand = {st: fu.stimulus_matrix(st, TEST_IDS, "test")[0] for st in stims}
    n_test = next(iter(Fte.values())).shape[0]

    # ---- refit every scorer and fuse
    M = np.zeros((n_test, len(TEST_IDS)), dtype=np.float64)
    used = 0
    for w, spec in zip(weights, specs):
        es, st, _d = spec.split(":")
        c = fu.Channel(spec).fit(Ftr[es], Etr[st])
        S = c.score(Fte[es], Ecand[st])
        M += w * tf(S)
        used += 1
    assert used == len(specs), f"{used} of {len(specs)} scorers applied"
    M = P.apply_postprocess(M, fz.get("postprocess", "none"))

    order = np.argsort(-M, axis=1)                    # best candidate first
    ranked = np.asarray(TEST_IDS, dtype=np.int64)[order]
    truth = cio.load_test_labels(sub, reason="independent end-to-end verification "
                                            "of the frozen pipeline")
    return ranked, np.asarray(truth)


def score_submission(ranked: np.ndarray, truth: np.ndarray) -> dict:
    """Metrics computed from the submission FILE, the way a grader would."""
    pos = np.array([int(np.where(ranked[i] == truth[i])[0][0]) + 1
                    for i in range(ranked.shape[0])])
    return {"top1": float(np.mean(pos <= 1)), "top5": float(np.mean(pos <= 5)),
            "mean_rank": float(np.mean(pos)), "median_rank": float(np.median(pos)),
            "mrr": float(np.mean(1.0 / pos)), "ranks": pos.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default=",".join(DEV_SUBJECTS))
    args = ap.parse_args()
    subs = [s.strip() for s in args.subjects.split(",") if s.strip()]
    OUTDIR.mkdir(parents=True, exist_ok=True)

    fz = json.loads((RESULTS_DIR / "compare_embed" / "_frozen_pipeline.json").read_text())
    print(f"frozen configuration: {fz['selected']}  ({len(fz['channels'])} scorers)")
    print("refitting every scorer from scratch; the channel cache is NOT used\n")

    out, allranks = {}, []
    for sub in subs:
        t0 = time.perf_counter()
        ranked, truth = build_and_rank(sub)
        p = OUTDIR / f"{sub}_ranked_candidate_list.npy"
        np.save(p, ranked.astype(np.int64))

        reloaded = np.load(p)                        # validate what was written
        validate_submission(reloaded)
        m = score_submission(reloaded, truth)
        out[sub] = {k: v for k, v in m.items() if k != "ranks"}
        out[sub]["submission_file"] = str(p.relative_to(ROOT))
        out[sub]["seconds"] = round(time.perf_counter() - t0, 1)
        allranks.append(np.array(m["ranks"]))
        print(f"  {sub}: top-1 {m['top1']*100:6.2f}%  top-5 {m['top5']*100:6.2f}%  "
              f"median {m['median_rank']:5.1f}   format OK   "
              f"({out[sub]['seconds']:.0f}s)", flush=True)

    r = np.concatenate(allranks)
    k1, k5, n = int((r <= 1).sum()), int((r <= 5).sum()), r.size
    pooled = {"top1": k1 / n, "top5": k5 / n, "k_top1": k1, "k_top5": k5,
              "n_trials": n, "mean_rank": float(np.mean(r)),
              "median_rank": float(np.median(r)), "mrr": float(np.mean(1.0 / r))}
    print(f"\n  POOLED  top-1 {k1/n*100:6.2f}%  top-5 {k5/n*100:6.2f}%  "
          f"({k5}/{n})  mean rank {np.mean(r):.2f}")

    # ---- compare against the published figure
    pub = json.loads((RESULTS_DIR / "compare_embed" / "_pipeline_test.json").read_text())
    pt = pub["pooled"]
    d1, d5 = abs(pooled["top1"] - pt["top1"]), abs(pooled["top5"] - pt["top5"])
    agree = d1 < 1e-9 and d5 < 1e-9
    print(f"\n  published by pipeline --test : top-1 {pt['top1']*100:.2f}%  "
          f"top-5 {pt['top5']*100:.2f}%")
    print(f"  this from-scratch rerun      : top-1 {pooled['top1']*100:.2f}%  "
          f"top-5 {pooled['top5']*100:.2f}%")
    print(f"  {'EXACT MATCH' if agree else f'DIFFERS by {d1:.2e} / {d5:.2e}'}")

    res = {"frozen_config": fz["selected"], "n_scorers": len(fz["channels"]),
           "method": "all scorers refitted from scratch on each subject's training "
                     "videos; test recordings read directly from tst/sub-XX_tst.npy; "
                     "submission files written, format-validated, then scored against "
                     "tst_stim/sub-XX_tst_stim.npy",
           "channel_cache_used": False,
           "per_subject": out, "pooled": pooled,
           "published_pooled": {"top1": pt["top1"], "top5": pt["top5"]},
           "matches_published": bool(agree)}
    pth = RESULTS_DIR / "compare_embed" / "_verify_on_test.json"
    pth.write_text(json.dumps(res, indent=2))
    print(f"  wrote {pth.relative_to(ROOT)}")
    print(f"  submission files in {OUTDIR.relative_to(ROOT)}/")
    return 0 if agree else 1


if __name__ == "__main__":
    raise SystemExit(main())
