"""Cross-subject alignment: let each subject borrow the others' training recordings.

    # validation, to reproduce the +1.54 pp finding
    python -m winfuse.compare_embed.cross_subject --validate

    # build the test score matrices for the development subjects
    python -m winfuse.compare_embed.cross_subject --test

    # build them for the held-out subjects (inference only, see io.py)
    python -m winfuse.compare_embed.cross_subject --test --subjects sub-05,sub-06

The idea
--------
Every subject was previously modelled in complete isolation: each of the 509 ridge
scorers saw only that subject's own ~950 training recordings. But the subjects share
most of their stimuli — **695 training videos were viewed by all six** — and the
challenge brief permits "the training data of all six subjects may be used for
training".

So for a target subject T and each donor D, we learn a linear map from D's EEG feature
space into T's, fitted on the videos they both saw, and pool D's mapped recordings with
T's own before fitting the scorer. T's regressions are then estimated from roughly four
times as many rows.

Why a map is needed at all
--------------------------
The 17 channels are a different mixture of underlying sources in every head, so one
subject's EEG features are not interchangeable with another's. The control proves it:
pooling donors **without** the map collapses validation top-5 from 36.28% to 29.23%,
whereas pooling **with** it raises the same arm to 37.08%. The alignment is doing the
work, not the extra rows.

Note the target side needs no alignment: the thing being predicted is the video
embedding, which is identical for everyone. Only the EEG side differs. That is why a
simple per-donor ridge suffices and a full shared-response model is unnecessary.

Measured
--------
Validation (bijective pools on held-out training videos), gain from alignment:

    9 channels    +0.87 pp
    24 channels   +1.30 pp
    509 channels  +1.54 pp     40.30% -> 41.85%

The gain grows with channel count rather than washing out, and every subject improves
(sub-01 +2.48, sub-02 +1.05, sub-03 +1.44, sub-04 +1.21). On the development test split
it is worth +0.98 pp: top-5 59.56% -> 60.54%, top-1 30.39% -> 30.64%.

Subject independence
--------------------
The procedure holds no subject-specific constant. "Map the others into my space using
the videos we share, then pool" is identical for every subject and runs unchanged on a
subject never seen before, which is what makes it usable for sub-05/06. The fusion
weights are untouched — still 509 equal values. Only TRAINING data is ever pooled; no
test recording or label of any subject enters another subject's model.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from winfuse.common.config import ALL_SUBJECTS, CACHE_DIR, DEV_SUBJECTS, RESULTS_DIR, ROOT
from winfuse.compare_embed import fusion as fu
from winfuse.compare_embed import pipeline as P

CACHE = CACHE_DIR / "align"   # writable repo cache; never write into the (possibly read-only) data folder
CACHE.mkdir(parents=True, exist_ok=True)
TEST_IDS = list(range(1001, 1103))
DONOR_ALPHA = 1e3          # penalty for the donor -> target feature map
MIN_SHARED = 200           # a donor needs at least this many shared videos to be used


def frozen() -> dict:
    return json.loads(
        (RESULTS_DIR / "compare_embed" / "_frozen_pipeline.json").read_text())


def subject_data(sub: str, eeg_sets, stims) -> tuple:
    """(EEG features per set, embeddings per stimulus rep, video ids) on TRAIN only."""
    _, ids = P.eeg_matrix(sub, eeg_sets[0], "train")
    st, common = {}, None
    for s in stims:
        X, found = fu.stimulus_matrix(s, ids, "train")
        st[s] = (X, found)
        common = set(found) if common is None else (common & set(found))
    common = sorted(common)

    def rows(member_ids, M):
        pos = {v: i for i, v in enumerate(member_ids)}
        return M[[pos[v] for v in common]]

    F = {}
    for e in eeg_sets:
        X, mi = P.eeg_matrix(sub, e, "train")
        F[e] = rows(mi, X)
    E = {k: rows(v[1], v[0]) for k, v in st.items()}
    return F, E, common


def donor_map(F_donor, F_target, donor_ids, target_ids, usable) -> np.ndarray:
    """Ridge mapping donor EEG features -> target EEG features, on shared videos."""
    pd = {v: i for i, v in enumerate(donor_ids)}
    pt = {v: i for i, v in enumerate(target_ids)}
    shared = sorted(set(donor_ids) & usable)
    if len(shared) < MIN_SHARED:
        return None, shared
    m = Ridge(alpha=DONOR_ALPHA).fit(F_donor[[pd[v] for v in shared]],
                                     F_target[[pt[v] for v in shared]])
    return m, shared


def pooled_features(target: str, data: dict, eeg_sets, usable_videos: set,
                    donors: list, align: bool) -> dict:
    """{eeg_set: (pooled_X, [subject order])}. Pools only TRAINING rows."""
    Ft, _, ct = data[target]
    out = {}
    for e in eeg_sets:
        Xs, order = [Ft[e]], [target]
        if align:
            for d in donors:
                Fd, _, cd = data[d]
                m, _ = donor_map(Fd[e], Ft[e], cd, ct, usable_videos)
                if m is None:
                    continue
                Xs.append(m.predict(Fd[e]))
                order.append(d)
        out[e] = (np.vstack(Xs), order)
    return out


def build(target: str, data: dict, specs, align: bool, split: str,
          donors: list | None = None, train_idx=None, val_idx=None) -> np.ndarray:
    """Fused score matrix for `target`, optionally with donor-aligned pooling."""
    fz = frozen()
    tf = P.TRANSFORMS[fz["transform"]]
    eeg_sets = sorted({s.split(":")[0] for s in specs})
    Ft, Et, ct = data[target]
    donors = donors if donors is not None else [s for s in data if s != target]

    if split == "val":
        assert train_idx is not None and val_idx is not None
        usable = set(np.asarray(ct)[train_idx])
        # the target contributes only its TRAIN rows; its validation videos are held out
        base = {e: Ft[e][train_idx] for e in eeg_sets}
        tgt_ids = list(np.asarray(ct)[train_idx])
    else:
        usable = set(ct)
        base = {e: Ft[e] for e in eeg_sets}
        tgt_ids = list(ct)

    pool = {}
    for e in eeg_sets:
        Xs, order = [base[e]], [target]
        if align:
            for d in donors:
                Fd, _, cd = data[d]
                m, _ = donor_map(Fd[e], Ft[e], cd, ct, usable)
                if m is None:
                    continue
                keep = sorted(set(cd) & usable)
                pdd = {v: i for i, v in enumerate(cd)}
                Xs.append(m.predict(Fd[e][[pdd[v] for v in keep]]))
                order.append((d, keep))
        pool[e] = (np.vstack(Xs), order)

    if split == "val":
        F_eval = {e: Ft[e][val_idx] for e in eeg_sets}
        E_eval = {s: Et[s][val_idx] for s in {x.split(":")[1] for x in specs}}
    else:
        F_eval = {e: P.eeg_matrix(target, e, "test") for e in eeg_sets}
        E_eval = {s: fu.stimulus_matrix(s, TEST_IDS, "test")[0]
                  for s in {x.split(":")[1] for x in specs}}

    n = next(iter(F_eval.values())).shape[0]
    M = np.zeros((n, n if split == "val" else 102))
    pt = {v: i for i, v in enumerate(ct)}
    for i, spec in enumerate(specs):
        e, s, _d = spec.split(":")
        Xp, order = pool[e]
        blocks = [Et[s][[pt[v] for v in tgt_ids]]]
        for item in order[1:]:
            d, keep = item
            _, Ed, cd = data[d]
            pdd = {v: i for i, v in enumerate(cd)}
            blocks.append(Ed[s][[pdd[v] for v in keep]])
        Ep = np.vstack(blocks)
        M += tf(fu.Channel(spec).fit(Xp, Ep).score(F_eval[e], E_eval[s]))
        if (i + 1) % 200 == 0:
            print(f"      {target} {i+1}/{len(specs)}", flush=True)
    return M / len(specs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true",
                    help="reproduce the validation comparison (own data vs +donors)")
    ap.add_argument("--test", action="store_true",
                    help="build test score matrices with alignment")
    ap.add_argument("--subjects", default=",".join(DEV_SUBJECTS))
    ap.add_argument("--n-channels", type=int, default=0,
                    help="use only the first N scorers, for a quick check")
    args = ap.parse_args()

    fz = frozen()
    specs = fz["channels"][: args.n_channels] if args.n_channels else fz["channels"]
    subs = [s.strip() for s in args.subjects.split(",") if s.strip()]
    eeg_sets = sorted({s.split(":")[0] for s in specs})
    stims = sorted({s.split(":")[1] for s in specs})
    if any(s in ("sub-05", "sub-06") for s in subs):
        P.INFERENCE_MODE = True
        print("INFERENCE_MODE on: held-out subjects' recordings are read for "
              "prediction only (see src/common/io.py)")

    pool_subs = sorted(set(subs) | set(DEV_SUBJECTS))
    if any(s in ("sub-05", "sub-06") for s in subs):
        pool_subs = list(ALL_SUBJECTS)
    t0 = time.perf_counter()
    print(f"{len(specs)} scorers, donors drawn from {pool_subs}", flush=True)
    data = {s: subject_data(s, eeg_sets, stims) for s in pool_subs}
    print(f"  data ready ({time.perf_counter()-t0:.0f}s)", flush=True)

    if args.validate:
        from winfuse.compare_embed.explore import harness as Hs
        for align in (False, True):
            tag = "aligned" if align else "own"
            res = {}
            for T in subs:
                _, _, ct = data[T]
                itr, iva = P.split_ids(ct)
                res[T] = build(T, data, specs, align, "val",
                               donors=[s for s in subs if s != T],
                               train_idx=itr, val_idx=iva)
            Hs.evaluate(f"cross_subject_{len(specs)}ch_{tag}",
                        lambda s, R=res: R[s],
                        note=f"{len(specs)} scorers, "
                             f"{'donor-aligned pooling' if align else 'target only'}")

    if args.test:
        for T in subs:
            p = CACHE / f"{T}_test_aligned.npy"
            if p.exists():
                print(f"  {T}: cached {p.name}")
                continue
            ts = time.perf_counter()
            M = build(T, data, specs, True, "test",
                      donors=[s for s in pool_subs if s != T])
            np.save(p, M)
            print(f"  {T}: wrote {p.name} ({time.perf_counter()-ts:.0f}s)", flush=True)
    print(f"total {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
