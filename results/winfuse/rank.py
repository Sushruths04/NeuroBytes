"""Rank the 102 candidate videos. Two modes, neither of which retrains the fusion.

    # (a) INSTANT -- no model fitting at all. Uses the shipped fused score matrices.
    python rank.py --from-scores --subjects sub-01,sub-02,sub-03,sub-04

    # (b) FULL -- fits this subject's own ridges, then applies the SAME general
    #     weights. Needed for any subject whose scores are not shipped (sub-05/06).
    python rank.py --fit --subjects sub-05 --data-root /path/to/challenge

Writes <sub>_ranked_candidate_list.npy: int, (102, 102), stimulus IDs 1001..1102,
most likely first, rows in the recordings' original file order.

What is general and what is not
-------------------------------
weights/fusion_weights.npy is SUBJECT-INDEPENDENT -- 509 equal weights, 1/509 each.
There is no subject-specific number anywhere in the fusion, so it transfers to an
unseen person unchanged. Measured cost of that transfer: leave-one-subject-out gives
top-5 58.58% against 59.56% when all four subjects are used to choose, i.e. +0.98 pp
of optimism.

The per-subject part is each ridge regression, and that is unavoidable: the 17
channels are a different mixture of sources in every head, so a ridge fitted on
sub-01 cannot read sub-05. Mode (b) fits those on the target subject's OWN training
recordings, which the brief permits.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEST_IDS = list(range(1001, 1103))


def sinkhorn(S, iters, tau):
    S = np.asarray(S, dtype=np.float64)
    Z = (S - S.mean(1, keepdims=True)) / np.maximum(S.std(1, keepdims=True), 1e-12)
    K = np.exp(np.clip(Z / tau, -700, 700))
    for _ in range(iters):
        K = K / np.maximum(K.sum(1, keepdims=True), 1e-300)
        K = K / np.maximum(K.sum(0, keepdims=True), 1e-300)
    return K


def to_ranked(M, cfg):
    if cfg.get("postprocess") == "sinkhorn":
        p = cfg.get("postprocess_params") or {"iters": 3, "tau": 0.5}
        M = sinkhorn(M, p["iters"], p["tau"])
    return np.asarray(TEST_IDS, dtype=np.int64)[np.argsort(-M, axis=1)]


def validate(a):
    assert a.shape[1] == 102, a.shape
    assert np.issubdtype(a.dtype, np.integer), a.dtype
    want = set(TEST_IDS)
    for i in range(a.shape[0]):
        assert set(a[i].tolist()) == want, f"row {i} is not a permutation of 1001..1102"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--from-scores", action="store_true",
                   help="instant: use the shipped fused score matrices")
    m.add_argument("--fit", action="store_true",
                   help="fit this subject's ridges, then apply the general weights")
    ap.add_argument("--subjects", required=True)
    ap.add_argument("--data-root", default="", help="for --fit: the challenge data root")
    ap.add_argument("--out", default=str(HERE / "ranked"))
    a = ap.parse_args()

    cfg = json.loads((HERE / "weights" / "config.json").read_text())
    w = np.load(HERE / "weights" / "fusion_weights.npy")
    specs = (HERE / "weights" / "channels.txt").read_text().split()
    assert len(w) == len(specs) == cfg["scorers"], "weights/channels/config disagree"
    nd = len(set(np.round(w, 12)))
    print(f"general weights: {len(w)} scorers, {nd} distinct value"
          f"{'' if nd == 1 else 's'}"
          f"{' (all equal -> nothing subject-specific)' if nd == 1 else ''}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    subs = [s.strip() for s in a.subjects.split(",") if s.strip()]

    if a.from_scores:
        for sub in subs:
            p = HERE / "scores" / f"{sub}_fused_scores.npy"
            if not p.exists():
                print(f"  {sub}: no shipped scores -- use --fit for this subject")
                continue
            r = to_ranked(np.load(p).astype(np.float64), cfg)
            validate(r)
            np.save(out / f"{sub}_ranked_candidate_list.npy", r)
            print(f"  {sub}: wrote {sub}_ranked_candidate_list.npy  {r.shape} {r.dtype}")
        return 0

    # ---- full path: needs the original source tree and the embedding cache
    root = Path(a.data_root or os.environ.get("WINFUSE_DATA_ROOT", ""))
    if not root or not (root / "trn").is_dir():
        print("ERROR: --fit needs --data-root pointing at a directory containing "
              "trn/, tst/, stimuli_train/, stimuli_test/")
        return 2
    os.environ["WINFUSE_DATA_ROOT"] = str(root)
    sys.path.insert(0, str(HERE / "pipeline_pkg"))
    try:
        from winfuse.compare_embed import fusion as fu
        from winfuse.compare_embed import pipeline as P
    except ImportError:
        print("ERROR: the importable package is missing. See README.md section "
              "'Running the full path'.")
        return 2

    tf = P.TRANSFORMS[cfg["score_transform"]]
    for sub in subs:
        eeg_sets = sorted({s.split(":")[0] for s in specs})
        stims = sorted({s.split(":")[1] for s in specs})
        F0, ids0 = P.eeg_matrix(sub, eeg_sets[0], "train")
        st_tr, common = {}, None
        for s in stims:
            X, f = fu.stimulus_matrix(s, ids0, "train")
            st_tr[s] = (X, f)
            common = set(f) if common is None else (common & set(f))
        common = sorted(common)

        def rows(mi, M, want):
            pos = {v: i for i, v in enumerate(mi)}
            return M[[pos[v] for v in want]]

        Ftr, Fte = {}, {}
        for e in eeg_sets:
            X, mi = P.eeg_matrix(sub, e, "train")
            Ftr[e] = rows(mi, X, common)
            Fte[e] = P.eeg_matrix(sub, e, "test")
        Etr = {s: rows(v[1], v[0], common) for s, v in st_tr.items()}
        Ec = {s: fu.stimulus_matrix(s, TEST_IDS, "test")[0] for s in stims}

        M = np.zeros((next(iter(Fte.values())).shape[0], 102))
        for wi, spec in zip(w, specs):
            es, stm, _ = spec.split(":")
            M += wi * tf(fu.Channel(spec).fit(Ftr[es], Etr[stm]).score(Fte[es], Ec[stm]))
        np.save(out / f"{sub}_fused_scores.npy", M.astype(np.float32))
        r = to_ranked(M, cfg)
        validate(r)
        np.save(out / f"{sub}_ranked_candidate_list.npy", r)
        print(f"  {sub}: wrote {sub}_ranked_candidate_list.npy  {r.shape} {r.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
