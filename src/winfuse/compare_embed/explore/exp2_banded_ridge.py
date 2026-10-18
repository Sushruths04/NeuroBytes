"""Experiment 2: banded (group-wise) ridge regression.

VALIDATION ONLY. Nothing here reads tst/ or tst_stim/. Judged on held-out TRAINING
videos through the shared harness, so numbers are comparable to the frozen baseline.

The question
------------
The current pipeline builds many SEPARATE scorers (one EEG set x one stimulus rep,
each with a single ridge alpha over all 150 PCA components) and fuses them by
reciprocal-rank averaging with equal weights (`rrf+hub`). Banded ridge instead
CONCATENATES several feature blocks into one regression and gives each block its own
penalty. Standard in voxel-wise encoding work; the claim to test is whether learning
per-block shrinkage jointly beats fusing separately-fit scorers after the fact.

Two blockings are implemented:

  (a) EEG-side banding   several latency windows (win_50_200, win_100_300,
                         win_50_400, win_200_400) for ONE stimulus representation,
                         a separate alpha per window block.
  (b) Stimulus-side banding  several stimulus representations for ONE EEG window,
                         a separate per-rep penalty.

The banded solve (EEG side)
---------------------------
A per-block ridge penalty alpha_b is exactly equivalent to scaling block b's design
columns by 1/sqrt(alpha_b) and solving a single ridge with unit penalty:

    argmin_W || Y - [X1 X2 ...] W ||^2 + sum_b alpha_b ||W_b||^2
  = argmin_W || Y - [X1/s1 X2/s2 ...] W' ||^2 + ||W'||^2,   s_b = sqrt(alpha_b)

so we reuse sklearn Ridge(alpha=1.0). The per-block scalings are chosen by a coarse
coordinate search scored by inner cross-validated VALIDATION retrieval MRR on the
TRAINING portion only.

Ranking space (critical, matches fusion.py)
--------------------------------------------
Decoding only. The ridge predicts STANDARDISED embeddings (ys = StandardScaler on the
target embedding); candidates are standardised with the same train-fitted scaler
before the cosine. Ranking in raw embedding space is a known ~2x top-5 regression.

Subject independence
--------------------
Everything is fitted per subject on THAT subject's own TRAINING videos. The procedure
(the grid, the coordinate search, the PCA/scaler/ridge recipe) is identical for every
subject and applies unchanged to a subject never seen before. Nothing is fitted across
subjects.

Run:
    .venv-embed/bin/python -m winfuse.compare_embed.explore.exp2_banded_ridge
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from winfuse.common.config import DEV_SUBJECTS, RESULTS_DIR, SEED
from winfuse.compare_embed import fusion as fu
from winfuse.compare_embed import pipeline as P
from winfuse.compare_embed.explore import harness as H

EEG_PCA = 150          # same as fusion.Channel decoding
# coarse per-block alpha grid for the coordinate search
BLOCK_ALPHA_GRID = np.logspace(0, 6, 7)  # 1, 10, ... 1e6
N_INNER_FOLDS = 3

# strong stimulus reps so iterations take minutes
STIM_REPS = ["resnet50@layer_layer3",
             "alexnet_layers@layer_fc6",
             "dinov2_base@layer_block09_meanpatch"]
# latency windows for EEG-side banding
EEG_WINDOWS = ["win_50_200", "win_100_300", "win_50_400", "win_200_400"]


# ----------------------------------------------------------------- block preparation
def _prep_block_pca(F_train: np.ndarray) -> tuple:
    """Fit PCA(<=150)+StandardScaler on a TRAIN block, return (pca, scaler)."""
    npc = int(min(EEG_PCA, F_train.shape[0] - 1, F_train.shape[1]))
    pca = PCA(n_components=npc, random_state=SEED).fit(F_train)
    xs = StandardScaler().fit(pca.transform(F_train))
    return pca, xs


def _apply_block(pca, xs, F: np.ndarray) -> np.ndarray:
    return xs.transform(pca.transform(F))


def _cos_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return An @ Bn.T


def _ranks_square(S: np.ndarray) -> np.ndarray:
    return np.array([int(np.sum(S[i] > S[i, i])) + 1 for i in range(S.shape[0])])


def _mrr_square(S: np.ndarray) -> float:
    r = _ranks_square(S)
    return float(np.mean(1.0 / r))


# -------------------------------------------------------------------- banded solve
class BandedDecoder:
    """Decoding (EEG blocks -> standardised embedding) with per-block ridge penalty.

    Blocks are concatenated after per-block PCA+scaling; each block is divided by
    sqrt(alpha_b); a single Ridge(alpha=1) is solved; candidates are ranked by cosine
    in the CENTRED embedding space (ys.transform), exactly as fusion.Channel does.
    """

    def __init__(self) -> None:
        self.blocks: list[tuple] = []     # (pca, xs) per block, fitted on TRAIN
        self.ys: StandardScaler | None = None
        self.model: Ridge | None = None
        self.scalings: np.ndarray | None = None

    def fit_blocks(self, F_blocks_train: list[np.ndarray], E_train: np.ndarray) -> "BandedDecoder":
        self.blocks = [_prep_block_pca(Fb) for Fb in F_blocks_train]
        self.ys = StandardScaler().fit(E_train)
        return self

    def _design(self, F_blocks: list[np.ndarray], scalings: np.ndarray) -> np.ndarray:
        cols = []
        for (pca, xs), Fb, s in zip(self.blocks, F_blocks, scalings):
            cols.append(_apply_block(pca, xs, Fb) / np.sqrt(s))
        return np.concatenate(cols, axis=1)

    def fit_ridge(self, F_blocks_train: list[np.ndarray], E_train: np.ndarray,
                  scalings: np.ndarray) -> "BandedDecoder":
        self.scalings = np.asarray(scalings, dtype=np.float64)
        X = self._design(F_blocks_train, self.scalings)
        Yt = self.ys.transform(E_train)
        self.model = Ridge(alpha=1.0).fit(X, Yt)
        return self

    def score(self, F_blocks_trials: list[np.ndarray], E_cand: np.ndarray) -> np.ndarray:
        X = self._design(F_blocks_trials, self.scalings)
        q = self.model.predict(X)                 # predicted standardised embedding
        return _cos_rows(q, self.ys.transform(E_cand))


def choose_scalings(F_blocks_train: list[np.ndarray], E_train: np.ndarray,
                    verbose: bool = False) -> np.ndarray:
    """Coordinate search over per-block alpha, scored by inner-CV retrieval MRR.

    TRAIN portion only. For each candidate alpha vector we refit the banded ridge on
    the inner-train fold and measure cosine retrieval MRR on the inner-held-out fold
    (candidates = the held-out videos themselves, truth on the diagonal). This matches
    the actual objective (retrieval), not plain regression MSE.
    """
    k = len(F_blocks_train)
    kf = KFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED)
    splits = list(kf.split(np.arange(E_train.shape[0])))

    def eval_scalings(scal: np.ndarray) -> float:
        mrrs = []
        for tr, te in splits:
            dec = BandedDecoder()
            dec.fit_blocks([Fb[tr] for Fb in F_blocks_train], E_train[tr])
            dec.fit_ridge([Fb[tr] for Fb in F_blocks_train], E_train[tr], scal)
            S = dec.score([Fb[te] for Fb in F_blocks_train], E_train[te])
            mrrs.append(_mrr_square(S))
        return float(np.mean(mrrs))

    # start at the single global alpha that is best on the same inner CV (coarse)
    best_global, best_s = BLOCK_ALPHA_GRID[0], -np.inf
    for a in BLOCK_ALPHA_GRID:
        s = eval_scalings(np.full(k, a))
        if s > best_s:
            best_global, best_s = a, s
    scal = np.full(k, best_global, dtype=np.float64)
    cur = best_s
    if verbose:
        print(f"      global-alpha start {best_global:.3g} -> inner MRR {cur:.4f}")

    # coordinate ascent, up to 2 sweeps
    for sweep in range(2):
        improved = False
        for b in range(k):
            best_v, best_bs = scal[b], cur
            for a in BLOCK_ALPHA_GRID:
                if a == scal[b]:
                    continue
                trial = scal.copy()
                trial[b] = a
                s = eval_scalings(trial)
                if s > best_bs + 1e-9:
                    best_v, best_bs = a, s
            if best_v != scal[b]:
                scal[b], cur = best_v, best_bs
                improved = True
        if verbose:
            print(f"      sweep {sweep+1}: alphas={scal.tolist()} inner MRR {cur:.4f}")
        if not improved:
            break
    return scal


# -------------------------------------------------------------- data loading helpers
def _common_ids(sub: str, eeg_sets: list[str], stims: list[str]):
    """Load EEG blocks and stimulus blocks aligned on common video ids (TRAIN)."""
    F_raw = {e: P.eeg_matrix(sub, e, "train") for e in eeg_sets}
    base_ids = F_raw[eeg_sets[0]][1]
    stim_raw, common = {}, None
    for st in stims:
        X, found = fu.stimulus_matrix(st, base_ids, "train")
        stim_raw[st] = (X, found)
        common = set(found) if common is None else (common & set(found))
    common = sorted(set(base_ids) & common)

    def rows(ids_have, M, want):
        pos = {v: i for i, v in enumerate(ids_have)}
        return M[[pos[v] for v in want]]

    F = {e: rows(F_raw[e][1], F_raw[e][0], common) for e in eeg_sets}
    E = {st: rows(stim_raw[st][1], stim_raw[st][0], common) for st in stims}
    return F, E, common


# --------------------------------------------------- (a) EEG-side banding arms
def banded_eeg_score_fn(eeg_sets: list[str], stim: str):
    """Banded ridge over EEG latency windows (blocks), single stim rep as target."""
    def fn(sub: str) -> np.ndarray:
        F, E, common = _common_ids(sub, eeg_sets, [stim])
        itr, iva = P.split_ids(common)
        F_blocks = [F[e] for e in eeg_sets]
        E_full = E[stim]
        F_tr = [Fb[itr] for Fb in F_blocks]
        E_tr = E_full[itr]
        scal = choose_scalings(F_tr, E_tr)
        dec = BandedDecoder().fit_blocks(F_tr, E_tr).fit_ridge(F_tr, E_tr, scal)
        F_va = [Fb[iva] for Fb in F_blocks]
        return dec.score(F_va, E_full[iva])
    return fn


def separate_eeg_score_fn(eeg_sets: list[str], stim: str):
    """Matched arm: k separate decoding Channels (each window x stim), rrf+hub fused."""
    tf = P.TRANSFORMS["rrf+hub"]

    def fn(sub: str) -> np.ndarray:
        F, E, common = _common_ids(sub, eeg_sets, [stim])
        itr, iva = P.split_ids(common)
        mats = []
        for e in eeg_sets:
            c = fu.Channel(f"{e}:{stim}:dec").fit(F[e][itr], E[stim][itr])
            mats.append(tf(c.score(F[e][iva], E[stim][iva])))
        return sum(mats) / len(mats)
    return fn


# --------------------------------------------------- (b) stimulus-side banding arms
def banded_stim_score_fn(eeg_set: str, stims: list[str]):
    """Stimulus-side banding: several embedding reps predicted from one EEG window.

    The reps are TARGETS, so per-block penalties act by weighting each rep's
    contribution to the ranking. Each rep keeps its own centred-space cosine (the
    non-negotiable ranking space), and the per-rep weights are chosen jointly by the
    same inner-CV retrieval-MRR coordinate search. This is 'one joint regression with
    per-block penalties' in the sense that the EEG design is shared and the per-rep
    weighting is learned together rather than fixed at equal.
    """
    def _z(S):
        return (S - S.mean(1, keepdims=True)) / np.maximum(S.std(1, keepdims=True), 1e-12)

    def fn(sub: str) -> np.ndarray:
        F, E, common = _common_ids(sub, [eeg_set], stims)
        itr, iva = P.split_ids(common)
        Feeg = F[eeg_set]
        k = len(stims)

        kf = KFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED)
        splits = list(kf.split(itr))
        # precompute per-rep z-scored cosine matrices per inner fold (fit inner-train)
        fold_mats = []
        for tr, te in splits:
            a_tr, a_te = itr[tr], itr[te]
            mats = []
            for st in stims:
                c = fu.Channel(f"{eeg_set}:{st}:dec").fit(Feeg[a_tr], E[st][a_tr])
                mats.append(_z(c.score(Feeg[a_te], E[st][a_te])))
            fold_mats.append(mats)

        def eval_w(w):
            ms = []
            for mats in fold_mats:
                M = sum(w[si] * mats[si] for si in range(k))
                ms.append(_mrr_square(M))
            return float(np.mean(ms))

        w = np.ones(k)
        cur = eval_w(w)
        wgrid = [0.0, 0.5, 1.0, 2.0, 4.0]
        for _ in range(2):
            for b in range(k):
                best_v, best_s = w[b], cur
                for v in wgrid:
                    if v == w[b]:
                        continue
                    t = w.copy(); t[b] = v
                    if t.sum() <= 0:
                        continue
                    s = eval_w(t)
                    if s > best_s + 1e-9:
                        best_v, best_s = v, s
                if best_v != w[b]:
                    w[b], cur = best_v, best_s

        # final: fit each rep on full itr, score iva, fuse with learned weights
        out = []
        for si, st in enumerate(stims):
            c = fu.Channel(f"{eeg_set}:{st}:dec").fit(Feeg[itr], E[st][itr])
            out.append(_z(c.score(Feeg[iva], E[st][iva])))
        return sum(w[si] * out[si] for si in range(k))
    return fn


def separate_stim_score_fn(eeg_set: str, stims: list[str]):
    """Matched arm for stimulus-side: k separate dec Channels, rrf+hub equal fused."""
    tf = P.TRANSFORMS["rrf+hub"]

    def fn(sub: str) -> np.ndarray:
        F, E, common = _common_ids(sub, [eeg_set], stims)
        itr, iva = P.split_ids(common)
        Feeg = F[eeg_set]
        mats = []
        for st in stims:
            c = fu.Channel(f"{eeg_set}:{st}:dec").fit(Feeg[itr], E[st][itr])
            mats.append(tf(c.score(Feeg[iva], E[st][iva])))
        return sum(mats) / len(mats)
    return fn


# ------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default=",".join(DEV_SUBJECTS))
    ap.add_argument("--stim", default="resnet50@layer_layer3",
                    help="single stim rep for EEG-side banding")
    args = ap.parse_args()
    subs = [s.strip() for s in args.subjects.split(",") if s.strip()]

    base = 0.40302287581699353        # frozen baseline pooled val top-5 (reference)

    # ---- (a) EEG-side banding -----------------------------------------------------
    stim = args.stim
    print(f"\n=== (a) EEG-side banding: windows {EEG_WINDOWS} x [{stim}] ===")
    print("  --- banded ---")
    r_band = H.evaluate(
        "02_banded_eeg_windows",
        banded_eeg_score_fn(EEG_WINDOWS, stim), subjects=subs,
        note=(f"banded ridge over EEG latency windows {EEG_WINDOWS} for {stim}; "
              f"per-window alpha by inner-CV retrieval-MRR coordinate search"))
    print("  --- matched separate scorers (rrf+hub equal) ---")
    r_sep = H.evaluate(
        "02_separate_eeg_windows",
        separate_eeg_score_fn(EEG_WINDOWS, stim), subjects=subs,
        note=(f"MATCHED ARM: {len(EEG_WINDOWS)} separate dec scorers over the same "
              f"windows for {stim}, fused by rrf+hub equal weights"))

    # ---- (b) stimulus-side banding ------------------------------------------------
    eeg_win = "win_50_400"
    print(f"\n=== (b) stimulus-side banding: [{eeg_win}] x {STIM_REPS} ===")
    print("  --- banded (per-rep weight by inner-CV retrieval MRR) ---")
    r_band2 = H.evaluate(
        "02_banded_stim_reps",
        banded_stim_score_fn(eeg_win, STIM_REPS), subjects=subs,
        note=(f"stimulus-side banding: {len(STIM_REPS)} reps predicted from "
              f"{eeg_win}, per-rep weight by inner-CV retrieval-MRR coordinate search"))
    print("  --- matched separate scorers (rrf+hub equal) ---")
    r_sep2 = H.evaluate(
        "02_separate_stim_reps",
        separate_stim_score_fn(eeg_win, STIM_REPS), subjects=subs,
        note=(f"MATCHED ARM: {len(STIM_REPS)} separate dec scorers ({eeg_win} x each "
              f"rep), fused by rrf+hub equal weights"))

    # ---- summary ------------------------------------------------------------------
    def t5(r):
        return r["pooled_with_bijection"]["top5"]

    def t1(r):
        return r["pooled_with_bijection"]["top1"]

    print("\n" + "=" * 72)
    print(f"  {'arm':40} {'val top-5':>12} {'val top-1':>12}")
    print("  " + "-" * 68)
    rows = [
        ("baseline (frozen, full pipeline)", base, 0.1513888888888889),
        ("(a) banded EEG windows", t5(r_band), t1(r_band)),
        ("(a) matched separate windows", t5(r_sep), t1(r_sep)),
        ("(b) banded stim reps", t5(r_band2), t1(r_band2)),
        ("(b) matched separate stim reps", t5(r_sep2), t1(r_sep2)),
    ]
    for name, v5, v1 in rows:
        print(f"  {name:40} {v5*100:11.2f}% {v1*100:11.2f}%")
    print("=" * 72)

    summary = {
        "baseline_top5": base,
        "arms": {name: {"top5": v5, "top1": v1} for name, v5, v1 in rows},
        "per_subject": {
            "banded_eeg_windows": r_band["per_subject"],
            "separate_eeg_windows": r_sep["per_subject"],
            "banded_stim_reps": r_band2["per_subject"],
            "separate_stim_reps": r_sep2["per_subject"],
        },
        "eeg_banding_beat_matched": bool(t5(r_band) > t5(r_sep)),
        "stim_banding_beat_matched": bool(t5(r_band2) > t5(r_sep2)),
        "eeg_banding_beat_baseline": bool(t5(r_band) > base),
        "stim_banding_beat_baseline": bool(t5(r_band2) > base),
    }
    p = RESULTS_DIR / "compare_embed" / "explore" / "02_banded_summary.json"
    p.write_text(json.dumps(summary, indent=2))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
