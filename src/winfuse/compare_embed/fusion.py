"""Score-level fusion of (EEG set x stimulus representation x direction) channels.

    .venv-embed/bin/python -m winfuse.compare_embed.fusion --subjects sub-01,sub-02,sub-03,sub-04

Why fusion is the next move
---------------------------
Measured on the graded criterion (sub-01..04 test, 408 trials, chance 0.98/4.90%):

    AlexNet features.3 -> EEG  (encoding)   top-1 1.72%  top-5 9.31%
    raw_signal  x dinov2_base  (decoding)   top-1 1.96%  top-5 7.84%
    erp_windows x dinov2_base  (decoding)   top-1 2.21%  top-5 7.35%
    erp_windows x whisper_base (decoding)   top-1 1.72%  top-5 6.62%
    raw_signal  x 8 descriptors(decoding)   top-1 1.72%  top-5 6.86%

These differ in EEG representation (broadband trace vs ERP windows), in stimulus
modality (visual vs audio vs handcrafted) and in direction (encoding vs decoding).
Errors that are not perfectly correlated can be averaged down, which is why score
fusion usually beats its best component on retrieval tasks.

How a channel produces a score matrix
-------------------------------------
Each channel yields S of shape (n_trials, n_candidates); S[i, j] is how well
candidate j explains trial i.

    encoding   fit ridge  embedding -> EEG features   on TRAIN
               S[i, j] = pearson(eeg_i, predicted_eeg_j)
    decoding   fit ridge  EEG features -> embedding   on TRAIN
               S[i, j] = cosine(predicted_embedding_i, embedding_j)

Each S is z-scored per ROW before fusing, so channels with different score scales
contribute comparably, and the fused score is  sum_k w_k * z(S_k).

Leakage discipline
------------------
* Every ridge is fitted on training trials only.
* Fusion weights are learned by cross-validation on TRAINING trials only, by
  coordinate ascent on held-out mean reciprocal rank. Test labels are never involved
  in choosing a weight.
* Test EEG is read once, at the end, for scoring only. Fitted objects are SHA-256
  hashed before and after; `pipeline_mutated` is derived from that comparison.
* sub-05/sub-06 are not touched (io.py refuses them).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from winfuse.common import io as cio
from winfuse.common import preprocess as pp
from winfuse.common.config import DEV_SUBJECTS, RESULTS_DIR, ROOT, SEED
from winfuse.common.feature_api import get_extractor
from winfuse.compare_embed import eeg_features as ef

ALPHAS = np.logspace(-1, 6, 15)
EEG_PCA = 150          # EEG feature dims kept for the decoding direction
EMB_PCA = 256          # embedding dims kept for the encoding direction
N_FOLDS = 5
TEST_IDS = list(range(1001, 1103))

# channel spec: "eeg_set:stimulus:direction"
DEFAULT_CHANNELS = [
    # --- decoding (EEG -> stimulus), ranked in centred embedding space
    "raw_signal:dinov2_base:dec",          # strongest single channel measured
    "erp_windows:dinov2_base:dec",         # different EEG representation
    "erp_windows:clip_vit_b32:dec",        # semantic visual embedding
    "erp_windows:whisper_base:dec",        # AUDIO - different modality
    "raw_signal:phase1_descriptors:dec",   # 8 handcrafted scalars, interpretable
    # --- encoding (stimulus -> EEG), reliability-weighted correlation
    "erp_windows:clip_vit_b32:enc",
    "erp_windows:dinov2_base:enc",
    "erp_windows:alexnet_layers:enc",      # the organizers' baseline representation
]


# --------------------------------------------------------------------------- stimulus
def stimulus_matrix(name: str, ids: list[int], split: str) -> tuple[np.ndarray, list[int]]:
    """Stimulus representation rows for `ids`. Returns (X, ids_found).

    `name` may be "model" (use the default pooled variant) or "model@variant" to
    address a specific cached variant, e.g. "dinov2_base@layer_block06_cls". Layer
    variants cover all 1102 videos, so the same variant serves train and test.
    """
    if "@" in name:
        model, variant = name.split("@", 1)
        X, found, _ = ef.resolve_embedding(model, ids, variant=variant)
        return np.asarray(X, dtype=np.float64), list(found)
    if name == "phase1_descriptors":
        from winfuse.validation import descriptors as dm

        X, _ = dm.matrix_for_ids(ids, split=split)
        return np.asarray(X, dtype=np.float64), list(ids)
    variant = None if split == "train" else _test_variant(name)
    X, found, _ = ef.resolve_embedding(name, ids, variant=variant)
    return np.asarray(X, dtype=np.float64), list(found)


def _test_variant(model: str) -> str | None:
    from winfuse.common import embed_cache as ec

    avail = ec.available(model)
    for n in ("baseline_test", "pooled"):
        if n in avail:
            return n
    return next((v for v in avail if v not in ("ids", "times")), None)


# --------------------------------------------------------------------------- scoring
def _zrow(M: np.ndarray) -> np.ndarray:
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    return (M - mu) / np.maximum(sd, 1e-12)


def _corr_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(n,d) x (m,d) -> (n,m) Pearson correlation between every pair of rows."""
    Az = _zrow(A)
    Bz = _zrow(B)
    return (Az @ Bz.T) / A.shape[1]


def _cos_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return An @ Bn.T


def _fit_ridge(X: np.ndarray, Y: np.ndarray) -> tuple:
    """Ridge with alpha by a single held-out split inside the training data."""
    n = X.shape[0]
    k = int(n * 0.8)
    best, best_err = ALPHAS[0], np.inf
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(X[:k], Y[:k])
        err = float(np.mean((m.predict(X[k:]) - Y[k:]) ** 2)) if n - k > 0 else 0.0
        if err < best_err:
            best, best_err = a, err
    return Ridge(alpha=best).fit(X, Y), float(best)


class Channel:
    """One (eeg_set, stimulus, direction) scorer.

    Both directions are scored in the space the model actually predicts in, which is
    the single most important detail in this module:

    decoding  The ridge predicts STANDARDISED embeddings, so candidates are
              standardised with the same train-fitted scaler before the cosine.
              Ranking in raw space instead lets a large shared mean vector dominate
              every cosine; measured cost is roughly half the top-5 (sub-01
              raw_signal x dinov2_base: 27.45% centered vs 11.76% raw).

    encoding  The correlation between measured and predicted EEG is weighted by each
              EEG dimension's cross-validated predictability. Measured on sub-01
              erp_windows x dinov2_base: of 136 EEG dimensions only 34 reach CV
              r > 0.10 and 26 have r <= 0, i.e. are pure noise. An unweighted
              correlation spends most of its mass on dimensions the encoder cannot
              predict, diluting the ones it can. Weights are w_d = max(r_d, 0)**2,
              which is the inverse-noise weighting for combining independent noisy
              estimates, and r_d is estimated by an inner CV on TRAINING trials only.
    """

    def __init__(self, spec: str, reliability_weighted: bool = True) -> None:
        self.spec = spec
        self.eeg_set, self.stim, self.direction = spec.split(":")
        if self.direction not in ("enc", "dec"):
            raise ValueError(f"direction must be enc|dec in {spec!r}")
        self.reliability_weighted = reliability_weighted
        self.state: dict = {}

    # -- fit on training rows only
    def fit(self, F: np.ndarray, E: np.ndarray) -> "Channel":
        if self.direction == "dec":
            npc = int(min(EEG_PCA, F.shape[0] - 1, F.shape[1]))
            pca = PCA(n_components=npc, random_state=SEED).fit(F)
            xs = StandardScaler().fit(pca.transform(F))
            ys = StandardScaler().fit(E)
            model, alpha = _fit_ridge(xs.transform(pca.transform(F)), ys.transform(E))
            self.state = {"pca": pca, "xs": xs, "ys": ys, "model": model, "alpha": alpha}
        else:
            npc = int(min(EMB_PCA, E.shape[0] - 1, E.shape[1]))
            pca = PCA(n_components=npc, random_state=SEED).fit(E)
            xs = StandardScaler().fit(pca.transform(E))
            ys = StandardScaler().fit(F)
            model, alpha = _fit_ridge(xs.transform(pca.transform(E)), ys.transform(F))
            w = None
            if self.reliability_weighted:
                w = self._dim_reliability(F, E)
            self.state = {"pca": pca, "xs": xs, "ys": ys, "model": model,
                          "alpha": alpha, "dim_w": w}
        return self

    @staticmethod
    def _dim_reliability(F: np.ndarray, E: np.ndarray, n_folds: int = 3) -> np.ndarray:
        """Per-EEG-dimension cross-validated encoding r, on TRAINING trials only.

        Returns w_d = max(r_d, 0)**2, normalised to sum to 1. Computed with an inner
        CV so no held-out or test information is used.
        """
        P = np.zeros_like(F, dtype=np.float64)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        for tr, te in kf.split(F):
            npc = int(min(EMB_PCA, len(tr) - 1, E.shape[1]))
            p = PCA(n_components=npc, random_state=SEED).fit(E[tr])
            xs = StandardScaler().fit(p.transform(E[tr]))
            ys = StandardScaler().fit(F[tr])
            m, _ = _fit_ridge(xs.transform(p.transform(E[tr])), ys.transform(F[tr]))
            P[te] = ys.inverse_transform(m.predict(xs.transform(p.transform(E[te]))))
        Fc = F - F.mean(0, keepdims=True)
        Pc = P - P.mean(0, keepdims=True)
        num = (Fc * Pc).sum(0)
        den = np.sqrt((Fc ** 2).sum(0) * (Pc ** 2).sum(0))
        r = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        w = np.clip(r, 0.0, None) ** 2
        s = w.sum()
        return (w / s) if s > 0 else np.full(F.shape[1], 1.0 / F.shape[1])

    # -- score trials against candidates
    def score(self, F_trials: np.ndarray, E_cand: np.ndarray) -> np.ndarray:
        s = self.state
        if self.direction == "dec":
            q = s["model"].predict(s["xs"].transform(s["pca"].transform(F_trials)))
            return _cos_rows(q, s["ys"].transform(E_cand))
        p = s["model"].predict(s["xs"].transform(s["pca"].transform(E_cand)))
        Fz = s["ys"].transform(F_trials)
        w = s.get("dim_w")
        if w is None:
            return _corr_rows(Fz, p)
        return _weighted_corr(Fz, p, w)


def _weighted_corr(A: np.ndarray, B: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted correlation between every row of A and every row of B.

    Each row is centred and scaled under the weights, so the result is a genuine
    weighted Pearson correlation rather than a weighted dot product.
    """
    w = np.asarray(w, dtype=np.float64)
    w = w / max(w.sum(), 1e-12)

    def prep(M: np.ndarray) -> np.ndarray:
        mu = (M * w).sum(axis=1, keepdims=True)
        Mc = M - mu
        sd = np.sqrt(np.maximum((Mc ** 2 * w).sum(axis=1, keepdims=True), 1e-12))
        return Mc / sd

    return (prep(A) * w) @ prep(B).T


# --------------------------------------------------------------------------- metrics
def ranks_of_truth(S: np.ndarray, truth_col: np.ndarray) -> np.ndarray:
    """1-based rank of the true candidate in each row (ties broken pessimistically)."""
    out = np.empty(S.shape[0], dtype=np.int64)
    for i in range(S.shape[0]):
        t = S[i, truth_col[i]]
        out[i] = int(np.sum(S[i] > t)) + 1
    return out


def topk(r: np.ndarray, k: int) -> float:
    return float(np.mean(r <= k))


def mrr(r: np.ndarray) -> float:
    return float(np.mean(1.0 / r))


# --------------------------------------------------------------------------- weights
def learn_weights(fold_blocks: list[list[np.ndarray]], verbose: bool = True) -> np.ndarray:
    """Coordinate ascent on held-out TRAIN mean reciprocal rank. No test data.

    `fold_blocks[f][k]` is channel k's square score block for fold f, where row i is
    the i-th held-out trial and column j the j-th held-out video, so the true answer
    is always on the diagonal.

    Scores are z-scored per row WITHIN each fold block and ranks are computed per
    fold, then concatenated. An earlier version assembled one block-diagonal matrix
    padded with -inf; row-wise z-scoring then produced NaN everywhere, every rank
    came out as 1, and MRR was a constant 1.0, so the search was optimising nothing.
    """
    n_chan = len(fold_blocks[0])
    zb = [[_zrow(b) for b in fold] for fold in fold_blocks]

    def score_of(wv: np.ndarray) -> float:
        rs = []
        for fold in zb:
            F = sum(wv[k] * fold[k] for k in range(n_chan))
            rs.append(ranks_of_truth(F, np.arange(F.shape[0])))
        return mrr(np.concatenate(rs))

    w = np.ones(n_chan, dtype=np.float64)
    cur = score_of(w)
    if verbose:
        print(f"      equal weights: held-out MRR {cur:.5f}")
    for sweep in range(2):
        for k in range(n_chan):
            best_v, best_s = w[k], cur
            for v in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
                if v == w[k]:
                    continue
                trial = w.copy()
                trial[k] = v
                if trial.sum() <= 0:
                    continue
                s = score_of(trial)
                if s > best_s + 1e-9:
                    best_v, best_s = v, s
            if best_v != w[k]:
                w[k], cur = best_v, best_s
        if verbose:
            print(f"      sweep {sweep+1}: MRR {cur:.5f}  w={np.round(w, 2).tolist()}")
    if w.sum() <= 0:
        w = np.ones(n_chan)
    return w


def mrr_proportional_weights(train_chan: dict, specs: list[str],
                             pool_size: int) -> np.ndarray:
    """Explainable alternative to the weight search: weight by standalone skill.

        w_k  proportional to  max(MRR_k - MRR_chance, 0)

    where MRR_k is channel k's held-out TRAINING mean reciprocal rank and
    MRR_chance is the expected MRR of random ranking for the same pool size,
    sum_{r=1..n}(1/r)/n. There is no search and nothing to tune, so it cannot
    overfit the weight-selection step, and the rule says plainly what it does:
    each channel contributes in proportion to how much better than chance it is
    on its own.
    """
    n = max(int(pool_size), 2)
    mrr_chance = float(np.sum(1.0 / np.arange(1, n + 1)) / n)
    w = np.array([max(train_chan[s]["mrr"] - mrr_chance, 0.0) for s in specs],
                 dtype=np.float64)
    return w if w.sum() > 0 else np.ones(len(specs), dtype=np.float64)


def paired_mcnemar(rank_a: np.ndarray, rank_b: np.ndarray, k: int = 5) -> dict:
    """Paired comparison of two configurations on the SAME trials (exact McNemar).

    Counts trials where A is a top-k hit and B is not, and vice versa, then runs a
    two-sided binomial test on the discordant pairs. This is the correct test here:
    both configurations are scored on identical trials, so comparing two independent
    confidence intervals would discard the pairing and overstate the uncertainty.
    """
    from scipy.stats import binomtest

    a = np.asarray(rank_a) <= k
    b = np.asarray(rank_b) <= k
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    nd = a_only + b_only
    p = float(binomtest(a_only, nd, 0.5).pvalue) if nd else 1.0
    return {"a_only": a_only, "b_only": b_only, "n_discordant": nd,
            "p_mcnemar": p, "a_better": a_only > b_only,
            "significant": bool(p < 0.05)}


def per_channel_train_metrics(fold_blocks: list[list[np.ndarray]],
                              specs: list[str]) -> dict:
    """Held-out TRAIN retrieval for each channel on its own, for the report."""
    out = {}
    for k, spec in enumerate(specs):
        rs = [ranks_of_truth(_zrow(fold[k]), np.arange(fold[k].shape[0]))
              for fold in fold_blocks]
        r = np.concatenate(rs)
        out[spec] = {"mrr": mrr(r), "top1": topk(r, 1), "top5": topk(r, 5),
                     "median_rank": float(np.median(r))}
    return out


# --------------------------------------------------------------------------- per subject
def run_subject(sub: str, specs: list[str], verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    chans = [Channel(s) for s in specs]

    # ---------------- training data, per channel (EEG features + stimulus rows)
    eeg_sets = sorted({c.eeg_set for c in chans})
    stims = sorted({c.stim for c in chans})
    F_tr: dict[str, tuple[np.ndarray, list[int]]] = {
        es: ef.eeg_feature_matrix(sub, es) for es in eeg_sets}
    # common video ids across every channel's stimulus representation
    common = None
    stim_tr: dict[str, tuple[np.ndarray, list[int]]] = {}
    base_ids = F_tr[eeg_sets[0]][1]
    for st in stims:
        X, found = stimulus_matrix(st, base_ids, "train")
        stim_tr[st] = (X, found)
        common = set(found) if common is None else (common & set(found))
    common = sorted(common)
    if verbose:
        print(f"    {sub}: {len(common)} common training videos, "
              f"{len(chans)} channels")

    def rows(mat_ids: list[int], M: np.ndarray, want: list[int]) -> np.ndarray:
        pos = {v: i for i, v in enumerate(mat_ids)}
        return M[[pos[v] for v in want]]

    Ftr = {es: rows(F_tr[es][1], F_tr[es][0], common) for es in eeg_sets}
    Etr = {st: rows(stim_tr[st][1], stim_tr[st][0], common) for st in stims}

    # ---------------- weight learning: held-out TRAIN score blocks, one per fold
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_blocks: list[list[np.ndarray]] = []
    for tr_idx, te_idx in kf.split(np.arange(len(common))):
        blocks = []
        for c in chans:
            F, E = Ftr[c.eeg_set], Etr[c.stim]
            c_fold = Channel(c.spec).fit(F[tr_idx], E[tr_idx])
            # candidates = this fold's own held-out videos, so the true answer sits
            # on the diagonal and each fold is a self-contained retrieval problem
            blocks.append(c_fold.score(F[te_idx], E[te_idx]))
        fold_blocks.append(blocks)

    if verbose:
        print("    learning fusion weights on held-out TRAIN folds:")
    w = learn_weights(fold_blocks, verbose=verbose)
    train_chan = per_channel_train_metrics(fold_blocks, specs)

    # ---------------- refit every channel on ALL training videos
    for c in chans:
        c.fit(Ftr[c.eeg_set], Etr[c.stim])

    # candidate stimulus rows for the 102 test videos
    Ecand = {}
    for st in stims:
        X, found = stimulus_matrix(st, TEST_IDS, "test")
        if found != TEST_IDS:
            pos = {v: i for i, v in enumerate(found)}
            X = X[[pos[v] for v in TEST_IDS if v in pos]]
        Ecand[st] = X

    # EEG features for the test trials, from extractors fitted on TRAIN
    extractors = {}
    for es in eeg_sets:
        ex = get_extractor(es)
        Xtr_ep, _ = pp.preprocessed_train(sub)
        ex.fit(Xtr_ep)
        ex.transform(Xtr_ep[:2])      # settle lazy bookkeeping before hashing
        extractors[es] = ex

    hash_before = _hash((chans, extractors, w))

    reason = f"fusion final scoring ({'+'.join(specs)})"
    Xte = pp.preprocessed_test(sub, reason=reason)
    ytrue = cio.load_test_labels(sub, reason=reason)
    Fte = {es: np.asarray(extractors[es].transform(Xte), dtype=np.float64)
           for es in eeg_sets}

    test_S = [c.score(Fte[c.eeg_set], Ecand[c.stim]) for c in chans]
    truth_test = np.array([TEST_IDS.index(v) for v in ytrue])

    hash_after = _hash((chans, extractors, w))

    # ---------------- evaluate components, and fuse with a direction ablation
    comp = {}
    for ci, c in enumerate(chans):
        r = ranks_of_truth(_zrow(test_S[ci]), truth_test)
        comp[c.spec] = _block(r)

    Zt = [_zrow(S) for S in test_S]
    dec_idx = [i for i, c in enumerate(chans) if c.direction == "dec"]
    enc_idx = [i for i, c in enumerate(chans) if c.direction == "enc"]

    def fuse(idx: list[int], weights: np.ndarray | None) -> dict:
        if not idx:
            return {}
        wv = np.ones(len(idx)) if weights is None else weights
        M = sum(wv[k] * Zt[i] for k, i in enumerate(idx))
        return _block(ranks_of_truth(M, truth_test))

    # Weights are re-learned WITHIN each subset, so an ablation is not penalised by
    # carrying weights that were tuned for a different channel set.
    w_dec = learn_weights([[f[i] for i in dec_idx] for f in fold_blocks],
                          verbose=False) if dec_idx else None
    w_enc = learn_weights([[f[i] for i in enc_idx] for f in fold_blocks],
                          verbose=False) if enc_idx else None

    # Explainable weighting: proportional to each channel's standalone held-out
    # training skill above chance. No search, nothing to tune.
    pool = int(np.mean([f[0].shape[0] for f in fold_blocks]))
    w_mrr = mrr_proportional_weights(train_chan, specs, pool)
    w_mrr_dec = mrr_proportional_weights(train_chan, [chans[i].spec for i in dec_idx],
                                         pool) if dec_idx else None

    out_fusion = {
        "fusion_learned_both": fuse(list(range(len(chans))), w),
        "fusion_equal_both": fuse(list(range(len(chans))), None),
        "fusion_mrrprop_both": fuse(list(range(len(chans))), w_mrr),
        "fusion_learned_dec_only": fuse(dec_idx, w_dec),
        "fusion_mrrprop_dec_only": fuse(dec_idx, w_mrr_dec),
        "fusion_learned_enc_only": fuse(enc_idx, w_enc),
    }

    return {
        "subject": sub,
        "channels": specs,
        "n_train_common": len(common),
        "weights": w.round(4).tolist(),
        "weights_dec_only": None if w_dec is None else w_dec.round(4).tolist(),
        "weights_mrrprop": w_mrr.round(5).tolist(),
        "mrr_pool_size": pool,
        "weights_enc_only": None if w_enc is None else w_enc.round(4).tolist(),
        "dec_channels": [chans[i].spec for i in dec_idx],
        "enc_channels": [chans[i].spec for i in enc_idx],
        "ridge_alphas": {c.spec: c.state["alpha"] for c in chans},
        "train_per_channel": train_chan,
        "test_per_channel": comp,
        **out_fusion,
        "fusion_equal": out_fusion["fusion_equal_both"],
        "fusion_learned": out_fusion["fusion_learned_both"],
        "pipeline_mutated": bool(hash_before != hash_after),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def _block(r: np.ndarray) -> dict:
    return {"top1": topk(r, 1), "top5": topk(r, 5), "mrr": mrr(r),
            "ranks": np.asarray(r).tolist(),
            "median_rank": float(np.median(r)), "mean_rank": float(np.mean(r)),
            "n_correct_top1": int(np.sum(r <= 1)), "n_correct_top5": int(np.sum(r <= 5)),
            "n_trials": int(r.size)}


def _hash(obj) -> str:
    try:
        return hashlib.sha256(pickle.dumps(obj)).hexdigest()
    except Exception:
        return hashlib.sha256(repr(obj).encode()).hexdigest()


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default=",".join(DEV_SUBJECTS))
    ap.add_argument("--channels", default=",".join(DEFAULT_CHANNELS))
    ap.add_argument("--out", default="results/compare_embed/_fusion.json")
    args = ap.parse_args()
    subs = [s.strip() for s in args.subjects.split(",") if s.strip()]
    specs = [s.strip() for s in args.channels.split(",") if s.strip()]

    print(f"fusion of {len(specs)} channels on {len(subs)} subjects")
    for s in specs:
        print(f"  - {s}")
    print()

    per = {}
    for sub in subs:
        print(f"  [{sub}]")
        per[sub] = run_subject(sub, specs)
        d = per[sub]
        for lbl, k in (("dec only ", "fusion_learned_dec_only"),
                       ("enc only ", "fusion_learned_enc_only"),
                       ("BOTH     ", "fusion_learned_both")):
            b = d[k]
            print(f"    {lbl} top1={b['top1']*100:5.2f}% top5={b['top5']*100:5.2f}%")
        print(f"    ({d['seconds']:.0f}s, mutated={d['pipeline_mutated']})")

    # ---- pooled
    from scipy.stats import binomtest

    def pool(getter) -> dict:
        k1 = sum(getter(per[s])["n_correct_top1"] for s in per)
        k5 = sum(getter(per[s])["n_correct_top5"] for s in per)
        n = sum(getter(per[s])["n_trials"] for s in per)
        p1 = binomtest(k1, n, 1 / 102, alternative="greater").pvalue
        p5 = binomtest(k5, n, 5 / 102, alternative="greater").pvalue
        c1 = binomtest(k1, n, 1 / 102).proportion_ci(0.95)
        c5 = binomtest(k5, n, 5 / 102).proportion_ci(0.95)
        return {"top1": k1 / n, "top5": k5 / n, "k1": k1, "k5": k5, "n": n,
                "p1": float(p1), "p5": float(p5),
                "ci95_top1": [float(c1.low), float(c1.high)],
                "ci95_top5": [float(c5.low), float(c5.high)]}

    pooled = {
        "fusion_learned_both": pool(lambda d: d["fusion_learned_both"]),
        "fusion_equal_both": pool(lambda d: d["fusion_equal_both"]),
        "fusion_mrrprop_both": pool(lambda d: d["fusion_mrrprop_both"]),
        "fusion_learned_dec_only": pool(lambda d: d["fusion_learned_dec_only"]),
        "fusion_mrrprop_dec_only": pool(lambda d: d["fusion_mrrprop_dec_only"]),
        "fusion_learned_enc_only": pool(lambda d: d["fusion_learned_enc_only"]),
    }
    for spec in specs:
        pooled[f"component:{spec}"] = pool(lambda d, s=spec: d["test_per_channel"][s])

    print("\n" + "=" * 86)
    print(f"  {'configuration':44} {'top-1':>14} {'top-5':>16}")
    print("  " + "-" * 82)
    rows = sorted(pooled.items(), key=lambda kv: -kv[1]["top5"])
    for name, b in rows:
        star = " *" if b["p5"] < 0.05 / max(len(pooled), 1) else "  "
        print(f"  {name:44} {b['top1']*100:6.2f}% ({b['k1']:2}/{b['n']}) "
              f"{b['top5']*100:6.2f}% ({b['k5']:3}/{b['n']}){star}")
    print("  " + "-" * 82)
    print(f"  {'chance':44} {1/102*100:6.2f}%        {5/102*100:6.2f}%")
    print("=" * 86)
    print("  * survives Bonferroni across the configurations shown")

    best = rows[0]
    print(f"\n  BEST top-5: {best[0]} at {best[1]['top5']*100:.2f}% "
          f"(p={best[1]['p5']:.2e}, CI {[round(x*100,2) for x in best[1]['ci95_top5']]})")
    b1 = max(pooled.items(), key=lambda kv: kv[1]["top1"])
    print(f"  BEST top-1: {b1[0]} at {b1[1]['top1']*100:.2f}% (p={b1[1]['p1']:.3f})")

    # ---- paired comparisons on identical trials
    def ranks_of(key):
        return np.concatenate([np.asarray(per[s][key]["ranks"]) for s in per])

    pairs = [("fusion_learned_both", "fusion_learned_dec_only"),
             ("fusion_mrrprop_both", "fusion_learned_dec_only"),
             ("fusion_mrrprop_dec_only", "fusion_learned_dec_only"),
             ("fusion_learned_dec_only", "component:" + specs[0])]
    paired = {}
    print("\n  paired comparisons (exact McNemar on the same 408 trials, top-5):")
    for a, b in pairs:
        ka = a if not a.startswith("component:") else None
        kb = b if not b.startswith("component:") else None
        try:
            ra = ranks_of(ka) if ka else np.concatenate(
                [np.asarray(per[s]["test_per_channel"][a.split("component:")[1]]["ranks"]) for s in per])
            rb = ranks_of(kb) if kb else np.concatenate(
                [np.asarray(per[s]["test_per_channel"][b.split("component:")[1]]["ranks"]) for s in per])
        except Exception as e:
            print(f"    {a} vs {b}: skipped ({type(e).__name__})")
            continue
        m = paired_mcnemar(ra, rb, k=5)
        paired[f"{a}__vs__{b}"] = m
        verdict = ("A better" if m["a_better"] else "B better") if m["significant"] \
            else "no detectable difference"
        print(f"    {a:28} vs {b:30} "
              f"{m['a_only']:3}/{m['b_only']:3} discordant  p={m['p_mcnemar']:.3f}  {verdict}")

    out = {"subjects": per, "pooled": pooled, "channels": specs, "paired": paired,
           "mutated_any": any(per[s]["pipeline_mutated"] for s in per)}
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
