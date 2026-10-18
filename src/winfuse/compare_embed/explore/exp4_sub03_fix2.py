"""PART B (cont.) - three more general per-subject remedies, same harness.

After the oscillatory-PCA suppression failed (exp4_sub03_fix: it slightly hurt
every subject, so sub-03's alpha excess is a correlate, not a fixable defect),
test the remaining candidates from the brief, all as general per-subject rules:

  A. robust per-trial rejection:   drop training trials whose window energy
     exceeds median + k*MAD (per subject), then fit on what remains. VAL is
     never pruned (we must score every val trial).
  B. per-channel robust scaling:   replace the window's implicit scaling with a
     per-channel median/IQR robust z-score fitted on TRAIN rows.
  C. covariance shrinkage:         shrink the EEG PCA/ridge input covariance via
     Ledoit-Wolf (sklearn) inside the decoder, stabilising the regression when
     the signal is low-SNR -- which is sub-03's situation.

Each is scored through H.evaluate against the SAME internal baseline
(exp4_sub03_fix_baseline, already written) and the frozen reference (40.30%).
Everything fitted on TRAIN only; VAL scored, never fitted; no test data.
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import winfuse.common.io as cio
from winfuse.common import preprocess as pp
from winfuse.common.config import SEED, SFREQ
from winfuse.compare_embed import eeg_features as ef
from winfuse.compare_embed import fusion as fu
from winfuse.compare_embed import pipeline as P
from winfuse.compare_embed.explore import harness as H
from winfuse.compare_embed.explore.exp4_sub03_fix import (
    WIN_SPECS, ERP_SPECS, _zrow, _window_from_epochs)


def _ridge_alpha(Xr, Yr):
    best, berr = 1.0, np.inf
    kk = int(Xr.shape[0] * 0.8)
    for a in np.logspace(-1, 6, 15):
        m = Ridge(alpha=a).fit(Xr[:kk], Yr[:kk])
        err = float(np.mean((m.predict(Xr[kk:]) - Yr[kk:]) ** 2))
        if err < berr:
            best, berr = a, err
    return best


def _decode(Ftr, Etr, Fva, Eva, shrink=False):
    npc = int(min(150, Ftr.shape[0] - 1, Ftr.shape[1]))
    pca = PCA(n_components=npc, random_state=SEED).fit(Ftr)
    Ztr = pca.transform(Ftr)
    if shrink:
        # whiten by Ledoit-Wolf-shrunk covariance of the PCA scores (TRAIN only)
        lw = LedoitWolf().fit(Ztr)
        cov = lw.covariance_
        ev, V = np.linalg.eigh(cov)
        ev = np.clip(ev, 1e-8, None)
        Wmat = V @ np.diag(ev ** -0.5) @ V.T
        xs_mean = Ztr.mean(0, keepdims=True)
        Ztr_w = (Ztr - xs_mean) @ Wmat
        xs = StandardScaler().fit(Ztr_w)
        ys = StandardScaler().fit(Etr)
        Xr, Yr = xs.transform(Ztr_w), ys.transform(Etr)
        a = _ridge_alpha(Xr, Yr)
        model = Ridge(alpha=a).fit(Xr, Yr)
        Zva_w = (pca.transform(Fva) - xs_mean) @ Wmat
        q = model.predict(xs.transform(Zva_w))
    else:
        xs = StandardScaler().fit(Ztr)
        ys = StandardScaler().fit(Etr)
        Xr, Yr = xs.transform(Ztr), ys.transform(Etr)
        a = _ridge_alpha(Xr, Yr)
        model = Ridge(alpha=a).fit(Xr, Yr)
        q = model.predict(xs.transform(pca.transform(Fva)))
    Ez = ys.transform(Eva)
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    en = Ez / np.maximum(np.linalg.norm(Ez, axis=1, keepdims=True), 1e-12)
    return qn @ en.T


def _robust_scale(Wtr, W):
    """Per-column robust z-score: (W - median_tr) / IQR_tr, fitted on TRAIN."""
    med = np.median(Wtr, axis=0, keepdims=True)
    q1 = np.percentile(Wtr, 25, axis=0, keepdims=True)
    q3 = np.percentile(Wtr, 75, axis=0, keepdims=True)
    iqr = np.maximum(q3 - q1, 1e-8)
    return (W - med) / iqr


def _trial_reject_mask(Wtr, k=5.0):
    """Boolean mask of TRAIN rows to KEEP: window energy within median+k*MAD."""
    e = (Wtr ** 2).mean(axis=1)
    med = np.median(e)
    mad = np.median(np.abs(e - med)) * 1.4826 + 1e-12
    return e <= (med + k * mad)


def make_score_fn(mode: str):
    cache = {}

    def score_fn(sub):
        if sub in cache:
            return cache[sub]
        Xep, _ = pp.preprocessed_train(sub)
        Xep = np.asarray(Xep, dtype=np.float64)
        ids = cio.subject_train_ids(sub)
        itr, iva = P.split_ids(ids)
        setr, seva = set(itr.tolist()), set(iva.tolist())

        blocks = []
        for es, st in WIN_SPECS:
            _, lo, hi = es.split("_")
            W = _window_from_epochs(Xep, int(lo), int(hi))
            E, found = fu.stimulus_matrix(st, ids, "train")
            pos = {v: i for i, v in enumerate(found)}
            keep = [i for i, v in enumerate(ids) if v in pos]
            W = W[keep]
            E = E[[pos[ids[i]] for i in keep]]
            idx_tr = np.array([j for j, i in enumerate(keep) if i in setr])
            idx_va = np.array([j for j, i in enumerate(keep) if i in seva])
            Wtr_raw = W[idx_tr]
            if mode == "robust_scale":
                Wsc = _robust_scale(Wtr_raw, W)
                S = _decode(Wsc[idx_tr], E[idx_tr], Wsc[idx_va], E[idx_va])
            elif mode == "trial_reject":
                m = _trial_reject_mask(Wtr_raw, k=5.0)
                S = _decode(Wtr_raw[m], E[idx_tr][m], W[idx_va], E[idx_va])
            elif mode == "shrinkage":
                S = _decode(Wtr_raw, E[idx_tr], W[idx_va], E[idx_va], shrink=True)
            else:
                raise ValueError(mode)
            blocks.append(_zrow(S))

        for es, st in ERP_SPECS:
            F, fids = ef.eeg_feature_matrix(sub, es)
            E, found = fu.stimulus_matrix(st, fids, "train")
            pos = {v: i for i, v in enumerate(found)}
            keep = [i for i, v in enumerate(fids) if v in pos]
            F = F[keep]
            E = E[[pos[fids[i]] for i in keep]]
            idx_tr = np.array([j for j, i in enumerate(keep) if i in setr])
            idx_va = np.array([j for j, i in enumerate(keep) if i in seva])
            S = _decode(F[idx_tr], E[idx_tr], F[idx_va], E[idx_va])
            blocks.append(_zrow(S))

        M = sum(blocks)
        cache[sub] = M
        return M

    return score_fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="trial_reject,robust_scale,shrinkage")
    args = ap.parse_args()
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        print(f"\n  fix mode = {mode}")
        H.evaluate(f"exp4_sub03_fix_{mode}", make_score_fn(mode),
                   note=f"general per-subject remedy '{mode}' on the window CNN "
                        f"decoders; erp decoders untouched; fitted on TRAIN only; "
                        f"runs unchanged on sub-05/06")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
