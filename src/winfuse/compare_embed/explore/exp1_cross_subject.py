"""Experiment 1: cross-subject shared-response modelling. VALIDATION ONLY.

    .venv-embed/bin/python -m winfuse.compare_embed.explore.exp1_cross_subject

The problem it attacks
----------------------
Every one of the frozen pipeline's 509 ridge regressions is fitted on a single
subject's ~950 training recordings and nothing else. But 695 training videos were
seen by ALL SIX subjects (measured), and every dev-subject pair shares 909-940. The
brief permits using all six subjects' training data. So a data-poor subject could, in
principle, borrow EEG evidence from the others.

Why the "common response space" collapses to a simple augmentation
-------------------------------------------------------------------
In the decoding direction the ridge maps  EEG_features -> video_embedding.  The
*target* of that regression -- the video embedding -- is already identical across
subjects (same video, same CLIP/ResNet vector). The only thing that differs across
subjects is the EEG side. So aligning subjects into a common response space is
exactly: map donor subject D's EEG into the target subject T's EEG feature space,
then add the mapped (EEG_D, embedding) pairs to T's own training set and fit ONE
pooled ridge. That is approach (c) in the brief, and it subsumes the SRM/Procrustes
framings here because the shared target is given, not learned.

The donor->target map is a ridge regression  EEG_D(v) -> EEG_T(v)  fitted only on
videos v that both D and T saw AND that are in T's TRAINING split (T's validation
videos are strictly held out). This makes the whole procedure subject-independent:
for a never-seen subject T=sub-05 one would learn the same donor->sub-05 maps from
shared TRAINING videos and pool exactly the same way; no label of T's is ever needed.

Variants
--------
baseline_repro   T's own train split only (should reproduce the frozen per-channel).
pooled_raw       pool donors' raw EEG with T's, NO alignment (control: expected to
                 hurt, because feature spaces are not comparable across subjects).
pooled_aligned   pool donors' EEG mapped through the donor->T ridge (the real idea).
aligned_w<k>     pooled_aligned but donor rows down-weighted by sample_weight=k.

All judged with the shared harness on bijective 102-pools of HELD-OUT TRAINING
videos. No tst/ or tst_stim/ is ever opened.
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from winfuse.common.config import DEV_SUBJECTS
from winfuse.compare_embed import fusion as fu
from winfuse.compare_embed import pipeline as P
from winfuse.compare_embed.explore import harness as H

# Strong, fast EEG sets and stimulus reps (from the brief + frozen top channels).
EEG_SETS = ["win_50_400", "win_50_300", "win_80_320"]
STIMS = ["resnet50@layer_layer3", "alexnet_layers@layer_fc6",
         "dinov2_base@layer_block09_meanpatch"]

# donor subjects available for alignment: all dev subjects except the one evaluated.
# (sub-05/06 would also be usable as donors in deployment, but their data is behind
# the leakage guard, so for this validation-only experiment donors are dev subjects.)
DONOR_POOL = list(DEV_SUBJECTS)

ALIGN_PCA = 150        # dims kept when learning the donor->target EEG map
ALIGN_ALPHA = 1e3      # ridge strength for the donor->target map
RIDGE_PCA = 150        # EEG dims kept in the decoding ridge (matches fusion.EEG_PCA)
EMB_PCA_CAP = 256


# ----------------------------------------------------------------- small helpers
def _row_index(ids):
    return {v: i for i, v in enumerate(ids)}


def _cos_rows(A, B):
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return An @ Bn.T


class _Cache:
    """Loads each subject's EEG matrix and the stimulus matrix once."""

    def __init__(self):
        self.eeg = {}      # (sub, eeg_set) -> (X, ids)
        self.stim = {}     # (stim, tuple(ids)) -> (E, ids_found) via fusion

    def eeg_matrix(self, sub, es):
        key = (sub, es)
        if key not in self.eeg:
            self.eeg[key] = P.eeg_matrix(sub, es, "train")
        return self.eeg[key]

    def stim_rows(self, stim, ids):
        """Embedding rows aligned to `ids` (all present for layer variants)."""
        E, found = fu.stimulus_matrix(stim, list(ids), "train")
        pos = _row_index(found)
        keep = [v for v in ids if v in pos]
        return np.asarray(E[[pos[v] for v in keep]], dtype=np.float64), keep


C = _Cache()


# ----------------------------------------------------------------- alignment
def learn_align_map(X_d, ids_d, X_t, ids_t, allowed_videos):
    """Ridge map from donor EEG -> target EEG on videos both saw AND allowed.

    `allowed_videos` is the set of target TRAINING-split videos (validation videos of
    the target are excluded here, so the map never sees them). Returns a callable
    that maps a donor EEG matrix (rows = any videos) into target feature space, plus
    the number of fitting videos.
    """
    pd, pt = _row_index(ids_d), _row_index(ids_t)
    common = [v for v in ids_t if v in pd and v in allowed_videos]
    if len(common) < 50:
        return None, len(common)
    Dd = X_d[[pd[v] for v in common]]
    Dt = X_t[[pt[v] for v in common]]
    # reduce donor side, standardise, ridge onto target features
    npc = int(min(ALIGN_PCA, Dd.shape[0] - 1, Dd.shape[1]))
    pca = PCA(n_components=npc, random_state=1234).fit(Dd)
    xs = StandardScaler().fit(pca.transform(Dd))
    reg = Ridge(alpha=ALIGN_ALPHA).fit(xs.transform(pca.transform(Dd)), Dt)

    def apply(Xq):
        return reg.predict(xs.transform(pca.transform(Xq)))

    return apply, len(common)


# ----------------------------------------------------------------- score functions
def _fit_decode_and_score(F_tr, E_tr, F_val, E_val, sample_weight=None):
    """Decode ridge EEG->embedding on (F_tr,E_tr); score val-vs-val by cosine.

    Mirrors fusion.Channel decoding: PCA+standardise EEG, standardise embedding,
    predict standardised embedding, cosine against standardised candidates.
    Returns (n_val, n_val) with truth on the diagonal.
    """
    npc = int(min(RIDGE_PCA, F_tr.shape[0] - 1, F_tr.shape[1]))
    pca = PCA(n_components=npc, random_state=1234).fit(F_tr)
    xs = StandardScaler().fit(pca.transform(F_tr))
    ys = StandardScaler().fit(E_tr)
    emb_pc = int(min(EMB_PCA_CAP, E_tr.shape[1]))
    reg = Ridge(alpha=1e3)
    Xr = xs.transform(pca.transform(F_tr))
    Yr = ys.transform(E_tr)
    if emb_pc < E_tr.shape[1]:
        epca = PCA(n_components=emb_pc, random_state=1234).fit(Yr)
        Yr_fit = epca.transform(Yr)
        reg.fit(Xr, Yr_fit, sample_weight=sample_weight)
        q = reg.predict(xs.transform(pca.transform(F_val)))
        cand = epca.transform(ys.transform(E_val))
    else:
        reg.fit(Xr, Yr, sample_weight=sample_weight)
        q = reg.predict(xs.transform(pca.transform(F_val)))
        cand = ys.transform(E_val)
    return _cos_rows(q, cand)


def _fuse(mats):
    """Average rrf+hub-transformed score matrices (the frozen transform)."""
    tf = P.TRANSFORMS["rrf+hub"]
    return sum(tf(m) for m in mats) / len(mats)


def _target_arrays(sub, es, stim):
    """(F, E, ids) for one subject on common EEG/embedding videos, train-ordered."""
    Xe, ide = C.eeg_matrix(sub, es)
    E, keep = C.stim_rows(stim, ide)
    pe = _row_index(ide)
    F = Xe[[pe[v] for v in keep]]
    return F, E, keep


def make_score_fn(mode, donor_weight=1.0):
    def score_fn(sub):
        mats = []
        donors = [d for d in DONOR_POOL if d != sub]
        for es in EEG_SETS:
            for stim in STIMS:
                F_t, E_t, ids_t = _target_arrays(sub, es, stim)
                itr, iva = P.split_ids(ids_t)
                tr_ids = [ids_t[i] for i in itr]
                allowed = set(tr_ids)             # target's own training videos only
                F_tr, E_tr = F_t[itr], E_t[itr]
                F_val, E_val = F_t[iva], E_t[iva]

                if mode == "baseline_repro":
                    mats.append(_fit_decode_and_score(F_tr, E_tr, F_val, E_val))
                    continue

                # pool donor rows (raw or aligned)
                F_parts, E_parts, w_parts = [F_tr], [E_tr], [np.ones(len(itr))]
                for d in donors:
                    Xd, idd = C.eeg_matrix(d, es)
                    Ed, keepd = C.stim_rows(stim, idd)   # donor embeddings (==target's)
                    pdd = _row_index(idd)
                    # donor rows restricted to videos in target's TRAIN split
                    cand_v = [v for v in keepd if v in allowed]
                    if len(cand_v) < 50:
                        continue
                    Fd_all = Xd[[pdd[v] for v in idd]]   # full donor matrix
                    if mode == "pooled_raw":
                        Fd_rows = Xd[[pdd[v] for v in cand_v]]
                    else:  # aligned
                        apply, n = learn_align_map(Xd, idd, F_t, ids_t, allowed)
                        if apply is None:
                            continue
                        Fd_rows = apply(Xd[[pdd[v] for v in cand_v]])
                    Ed_rows, _ = C.stim_rows(stim, cand_v)
                    F_parts.append(Fd_rows)
                    E_parts.append(Ed_rows)
                    w_parts.append(np.full(len(cand_v), donor_weight))
                F_pool = np.vstack(F_parts)
                E_pool = np.vstack(E_parts)
                sw = np.concatenate(w_parts)
                sw = None if np.allclose(sw, sw[0]) else sw
                mats.append(_fit_decode_and_score(F_pool, E_pool, F_val, E_val,
                                                  sample_weight=sw))
        return _fuse(mats)

    return score_fn


VARIANTS = {
    "baseline_repro": ("baseline_repro", 1.0),
    "pooled_raw": ("pooled_raw", 1.0),
    "pooled_aligned": ("pooled_aligned", 1.0),
    "aligned_w0.5": ("pooled_aligned", 0.5),
    "aligned_w0.25": ("pooled_aligned", 0.25),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    want = [v.strip() for v in args.variants.split(",") if v.strip()]

    base = H.frozen_baseline()["pooled_with_bijection"]["top5"]
    print(f"\nfrozen baseline (full 509-channel pipeline): val top-5 {base*100:.2f}%")
    print(f"this experiment uses {len(EEG_SETS)} EEG sets x {len(STIMS)} stims "
          f"= {len(EEG_SETS)*len(STIMS)} decoding channels, rrf+hub fused\n")

    results = {}
    for name in want:
        mode, dw = VARIANTS[name]
        print(f"[{name}] mode={mode} donor_weight={dw}")
        res = H.evaluate(f"01_cross_subject_{name}", make_score_fn(mode, dw),
                         note=f"cross-subject {mode}, donor_weight={dw}; "
                              f"EEG={EEG_SETS} STIM={STIMS}")
        results[name] = res

    print("\n" + "=" * 70)
    print(f"  {'variant':24} {'val top-5':>10} {'val top-1':>10}   vs 509-baseline")
    print("  " + "-" * 66)
    for name in want:
        r = results[name]["pooled_with_bijection"]
        print(f"  {name:24} {r['top5']*100:9.2f}% {r['top1']*100:9.2f}%")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
