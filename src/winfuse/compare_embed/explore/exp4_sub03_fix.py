"""PART B - fix. VALIDATION ONLY, scored through the shared harness.

Diagnosis (exp4_sub03_diagnose / exp4_sub03_regression):
  * sub-03's loss is reproduced from TRAINING data alone: 5-fold held-out
    TRAIN retrieval equal-fusion top-5 is 13.2% vs 21.0/19.9/14.8 for the
    other dev subjects, and the deficit is concentrated in the time-domain
    WINDOW CNN decoders (win_50_400 x alexnet_fc6 / resnet50_layer3), not in
    the erp/raw channels.
  * sub-03 has by far the most alpha-band (8-13 Hz) power: 33.5% of total vs
    13-25%, and it is elevated on EVERY channel (min-channel alpha 0.27 vs
    0.12-0.22). Alpha is a large stimulus-UNRELATED rhythm; in the time-domain
    window it adds variance exactly where the CNN decoders read, diluting the
    stimulus-locked component.

General per-subject remedy tested here
--------------------------------------
Suppress each subject's dominant stimulus-unrelated oscillatory variance in the
time-domain window features, by projecting out the leading principal components
of the per-subject, per-channel, trial-demeaned window matrix. The number of
components removed is chosen PER SUBJECT from that subject's own training data by
how concentrated its spectrum is in the alpha band -- more alpha => remove more.
Nothing about sub-03 is hard-coded; the rule reads a quantity every subject has
and would run unchanged on sub-05/06.

We score, through the SAME harness and the SAME bijective pools, both:
  * exp4_sub03_fix_baseline   -- window+erp decoder fusion, NO transform
  * exp4_sub03_fix_osc_pca    -- the same, WITH the per-subject suppression
so the effect is a within-experiment, apples-to-apples delta, and we also run
the frozen pipeline baseline (00_frozen_baseline, 40.30% pooled val top-5) for
the absolute reference.

Leakage discipline: everything is fitted on the subject's TRAIN split only; VAL
is scored, never fitted. No test data is touched (the harness forbids it).
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import winfuse.common.io as cio
from winfuse.common import preprocess as pp
from winfuse.common.config import SEED, SFREQ
from winfuse.compare_embed import fusion as fu
from winfuse.compare_embed.explore import harness as H

# compact, representative decoder set: the window CNN decoders that carry the
# sub-03 deficit, plus two erp decoders that do not, so a general fix must not
# wreck the healthy channels either.
WIN_SPECS = [
    ("win_50_400", "alexnet_layers@layer_fc6"),
    ("win_50_300", "alexnet_layers@layer_fc6"),
    ("win_50_400", "resnet50@layer_layer3"),
    ("win_80_320", "resnet50@layer_layer3"),
]
ERP_SPECS = [
    ("erp_windows", "dinov2_base"),
    ("erp_windows", "clip_vit_b32"),
]
ALPHA_LO, ALPHA_HI = 8.0, 13.0


def _zrow(M):
    mu = M.mean(1, keepdims=True)
    sd = M.std(1, keepdims=True)
    return (M - mu) / np.maximum(sd, 1e-12)


def _window_from_epochs(X, lo_ms, hi_ms):
    """(n,17,750) -> (n, 17*win) flattened window, matching P.window_eeg."""
    i0 = max(int(round(lo_ms / 1000.0 * SFREQ)), 0)
    i1 = min(int(round(hi_ms / 1000.0 * SFREQ)), X.shape[2])
    W = np.asarray(X[:, :, i0:i1], dtype=np.float64)
    return W.reshape(W.shape[0], -1)


def _alpha_fraction(X):
    """Per-subject mean alpha (8-13 Hz) power fraction, TRAIN epochs only."""
    X = np.asarray(X, dtype=np.float64)
    n, nch, nt = X.shape
    f, Pxx = welch(X.reshape(n * nch, nt), fs=SFREQ, nperseg=min(256, nt), axis=-1)
    Pxx = Pxx.reshape(n, nch, -1).mean(axis=(0, 1))
    tot = Pxx.sum()
    amask = (f >= ALPHA_LO) & (f < ALPHA_HI)
    return float(Pxx[amask].sum() / max(tot, 1e-12))


def _n_components_for(alpha_frac):
    """General per-subject rule: remove more oscillatory PCs when the subject's
    spectrum is more alpha-dominated. Baseline reference alpha ~0.13 (healthiest
    dev subject); map the EXCESS alpha above that to a component count.

    k = round( scale * max(alpha_frac - a0, 0) ), clipped to [0, kmax].

    a0, scale, kmax are fixed constants, not fitted to any subject's label. The
    only per-subject input is that subject's own training alpha fraction, which
    sub-05/06 also have. k=0 means 'do nothing', so a low-alpha subject is left
    untouched.
    """
    a0 = 0.15
    scale = 60.0
    kmax = 8
    k = int(round(scale * max(alpha_frac - a0, 0.0)))
    return int(min(k, kmax))


class _OscSuppressor:
    """Project out the leading k PCs of a per-subject window matrix (fitted on
    TRAIN rows only). k is chosen per subject from its training alpha fraction.
    Applied identically to TRAIN and VAL rows of that subject."""

    def __init__(self, k):
        self.k = k
        self.mean_ = None
        self.comp_ = None

    def fit(self, Wtr):
        self.mean_ = Wtr.mean(0, keepdims=True)
        if self.k <= 0:
            self.comp_ = None
            return self
        p = PCA(n_components=self.k, random_state=SEED).fit(Wtr - self.mean_)
        self.comp_ = p.components_                      # (k, d)
        return self

    def transform(self, W):
        Wc = W - self.mean_
        if self.comp_ is None:
            return Wc + self.mean_
        proj = (Wc @ self.comp_.T) @ self.comp_         # reconstruct the k-dim part
        return (Wc - proj) + self.mean_


def _decode_val(Ftr, Etr, Fva, Eva):
    """One decoder channel: fit ridge EEG->embedding on TRAIN, score VAL as a
    (n_val, n_val) cosine matrix with truth on the diagonal (uses fusion.Channel
    internals indirectly via a fresh Channel on a synthetic spec)."""
    # emulate Channel('x:y:dec') pipeline: PCA(EEG)->ridge->cosine in std emb space
    from sklearn.decomposition import PCA as _PCA
    npc = int(min(150, Ftr.shape[0] - 1, Ftr.shape[1]))
    pca = _PCA(n_components=npc, random_state=SEED).fit(Ftr)
    xs = StandardScaler().fit(pca.transform(Ftr))
    ys = StandardScaler().fit(Etr)
    # alpha by a single inner split, matching fusion._fit_ridge spirit
    Xr = xs.transform(pca.transform(Ftr))
    Yr = ys.transform(Etr)
    best, berr = 1.0, np.inf
    kk = int(Xr.shape[0] * 0.8)
    for a in np.logspace(-1, 6, 15):
        m = Ridge(alpha=a).fit(Xr[:kk], Yr[:kk])
        err = float(np.mean((m.predict(Xr[kk:]) - Yr[kk:]) ** 2))
        if err < berr:
            best, berr = a, err
    model = Ridge(alpha=best).fit(Xr, Yr)
    q = model.predict(xs.transform(pca.transform(Fva)))      # (n_val, emb)
    Ez = ys.transform(Eva)
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    en = Ez / np.maximum(np.linalg.norm(Ez, axis=1, keepdims=True), 1e-12)
    return qn @ en.T


def make_score_fn(use_fix: bool):
    cache = {}

    def score_fn(sub):
        if sub in cache:
            return cache[sub]
        Xep, _ = pp.preprocessed_train(sub)
        Xep = np.asarray(Xep, dtype=np.float64)
        ids = cio.subject_train_ids(sub)
        itr, iva = __import__("winfuse.compare_embed.pipeline", fromlist=["split_ids"]).split_ids(ids)

        alpha = _alpha_fraction(Xep)
        k = _n_components_for(alpha) if use_fix else 0

        blocks = []
        # ---- window decoders (the deficit carriers) ----
        for es, st in WIN_SPECS:
            _, lo, hi = es.split("_")
            W = _window_from_epochs(Xep, int(lo), int(hi))
            E, found = fu.stimulus_matrix(st, ids, "train")
            pos = {v: i for i, v in enumerate(found)}
            keep = [i for i, v in enumerate(ids) if v in pos]
            W = W[keep]
            E = E[[pos[ids[i]] for i in keep]]
            idx_tr = np.array([j for j, i in enumerate(keep) if i in set(itr.tolist())])
            idx_va = np.array([j for j, i in enumerate(keep) if i in set(iva.tolist())])
            sup = _OscSuppressor(k).fit(W[idx_tr])
            Wt = sup.transform(W)
            S = _decode_val(Wt[idx_tr], E[idx_tr], Wt[idx_va], E[idx_va])
            blocks.append(_zrow(S))
        # ---- erp decoders (healthy channels; left untransformed) ----
        from winfuse.compare_embed import eeg_features as ef
        for es, st in ERP_SPECS:
            F, fids = ef.eeg_feature_matrix(sub, es)
            E, found = fu.stimulus_matrix(st, fids, "train")
            pos = {v: i for i, v in enumerate(found)}
            keep = [i for i, v in enumerate(fids) if v in pos]
            F = F[keep]
            E = E[[pos[fids[i]] for i in keep]]
            kept_ids = [fids[i] for i in keep]
            setr, seva = set(itr.tolist()), set(iva.tolist())
            idx_tr = np.array([j for j, i in enumerate(keep) if i in setr])
            idx_va = np.array([j for j, i in enumerate(keep) if i in seva])
            S = _decode_val(F[idx_tr], E[idx_tr], F[idx_va], E[idx_va])
            blocks.append(_zrow(S))

        M = sum(blocks)
        cache[sub] = M
        return M

    return score_fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-ref", action="store_true",
                    help="also run the frozen pipeline baseline for the 40.30%% ref")
    args = ap.parse_args()

    if args.frozen_ref:
        print("  frozen pipeline reference:")
        H.frozen_baseline()

    print("\n  internal baseline (window+erp decoder fusion, NO transform):")
    H.evaluate("exp4_sub03_fix_baseline", make_score_fn(False),
               note="4 window CNN decoders + 2 erp decoders, equal fusion, no "
                    "oscillatory suppression; the matched control for the fix")

    print("\n  fix (per-subject oscillatory-variance suppression on window feats):")
    H.evaluate("exp4_sub03_fix_osc_pca", make_score_fn(True),
               note="same channels; project out leading k PCs of each subject's "
                    "trial-demeaned window matrix, k chosen from that subject's "
                    "own training alpha fraction (a0=0.15, scale=60, kmax=8); "
                    "general per-subject rule, runs unchanged on sub-05/06")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
