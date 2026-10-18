"""PART A (cont.) - the direct link to the pipeline: cross-validated EEG->embedding
regression quality per subject, plus a refined per-channel spectral analysis.

TRAINING DATA ONLY. Uses the same ridge machinery the pipeline uses
(fu.Channel), fitted on a subject's own training videos, scored by 5-fold CV
retrieval on held-out TRAINING videos. If sub-03's regressions are simply
weaker, that is the explanation; this quantifies it on the same footing as the
pipeline, with no test data.

Also computes per-channel alpha-band power fraction to see whether the excess
alpha seen in exp4_sub03_diagnose is global or carried by a few channels.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.signal import welch
from sklearn.model_selection import KFold

from winfuse.common import preprocess as pp
from winfuse.common.config import RESULTS_DIR, SEED, SFREQ
from winfuse.compare_embed import eeg_features as ef
from winfuse.compare_embed import fusion as fu

SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04"]
OUT = RESULTS_DIR / "compare_embed" / "explore"
OUT.mkdir(parents=True, exist_ok=True)

# a small, representative set of the pipeline's channels (decoding direction),
# so the CV retrieval numbers are comparable across subjects
PROBE_SPECS = [
    "win_50_400:alexnet_layers@layer_fc6:dec",
    "win_50_400:resnet50@layer_layer3:dec",
    "erp_windows:dinov2_base:dec",
    "raw_signal:dinov2_base:dec",
    "erp_windows:clip_vit_b32:dec",
]
N_FOLDS = 5


def _zrow(M):
    mu = M.mean(1, keepdims=True)
    sd = M.std(1, keepdims=True)
    return (M - mu) / np.maximum(sd, 1e-12)


def _ranks(S):
    return np.array([int(np.sum(S[i] > S[i, i])) + 1 for i in range(S.shape[0])])


def cv_retrieval(sub: str) -> dict:
    """5-fold held-out TRAIN retrieval for each probe channel + their equal fusion."""
    chans = [fu.Channel(s) for s in PROBE_SPECS]
    eeg_sets = sorted({c.eeg_set for c in chans})
    stims = sorted({c.stim for c in chans})
    F_tr = {es: ef.eeg_feature_matrix(sub, es) if not es.startswith("win_")
            else _win_matrix(sub, es) for es in eeg_sets}
    base_ids = F_tr[eeg_sets[0]][1]
    common = None
    stim_tr = {}
    for st in stims:
        X, found = fu.stimulus_matrix(st, base_ids, "train")
        stim_tr[st] = (X, found)
        common = set(found) if common is None else (common & set(found))
    common = sorted(common)

    def rows(mi, M, want):
        pos = {v: i for i, v in enumerate(mi)}
        return M[[pos[v] for v in want]]

    Ftr = {es: rows(F_tr[es][1], F_tr[es][0], common) for es in eeg_sets}
    Etr = {st: rows(stim_tr[st][1], stim_tr[st][0], common) for st in stims}

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    per_chan_ranks = {s: [] for s in PROBE_SPECS}
    fused_ranks = []
    for tr, te in kf.split(np.arange(len(common))):
        blocks = []
        for c in chans:
            F, E = Ftr[c.eeg_set], Etr[c.stim]
            cf = fu.Channel(c.spec).fit(F[tr], E[tr])
            S = cf.score(F[te], E[te])
            blocks.append(_zrow(S))
            per_chan_ranks[c.spec].append(_ranks(_zrow(S)))
        fused = sum(blocks)
        fused_ranks.append(_ranks(fused))

    def summ(rlist):
        r = np.concatenate(rlist)
        return {"top1": float(np.mean(r <= 1)), "top5": float(np.mean(r <= 5)),
                "mrr": float(np.mean(1.0 / r)), "median_rank": float(np.median(r))}

    out = {"per_channel": {s: summ(per_chan_ranks[s]) for s in PROBE_SPECS},
           "equal_fusion": summ(fused_ranks)}
    return out


def _win_matrix(sub, es):
    from winfuse.compare_embed import pipeline as P
    _, lo, hi = es.split("_")
    import winfuse.common.io as cio
    X = P.window_eeg(sub, int(lo), int(hi), "train")
    return X, cio.subject_train_ids(sub)


def per_channel_spectrum(sub: str) -> dict:
    X, _ = pp.preprocessed_train(sub)
    X = np.asarray(X, dtype=np.float64)
    n, nch, nt = X.shape
    f, Pxx = welch(X.reshape(n * nch, nt), fs=SFREQ, nperseg=min(256, nt), axis=-1)
    Pxx = Pxx.reshape(n, nch, -1).mean(axis=0)       # (nch, nfreq)
    tot = Pxx.sum(axis=1, keepdims=True)
    amask = (f >= 8.0) & (f < 13.0)
    alpha_frac_ch = (Pxx[:, amask].sum(axis=1) / tot.ravel())
    return {"alpha_frac_per_ch": alpha_frac_ch.round(4).tolist(),
            "alpha_frac_mean": float(alpha_frac_ch.mean()),
            "alpha_frac_max": float(alpha_frac_ch.max()),
            "alpha_frac_min": float(alpha_frac_ch.min())}


def main() -> int:
    res = {}
    for s in SUBJECTS:
        print(f"  [{s}] CV retrieval + spectrum ...", flush=True)
        res[s] = {"cv_retrieval": cv_retrieval(s),
                  "spectrum": per_channel_spectrum(s)}
    (OUT / "exp4_sub03_regression.json").write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 86)
    print("  CV held-out TRAIN retrieval (equal fusion of 5 probe channels)")
    print("=" * 86)
    print(f"  {'subject':10} {'top1':>8} {'top5':>8} {'mrr':>8} {'median':>8}")
    for s in SUBJECTS:
        m = res[s]["cv_retrieval"]["equal_fusion"]
        print(f"  {s:10} {m['top1']*100:7.2f}% {m['top5']*100:7.2f}% "
              f"{m['mrr']:7.4f} {m['median_rank']:7.1f}")

    print("\n  per-channel CV top-5 (held-out TRAIN):")
    print(f"  {'spec':44} " + "".join(f"{s:>10}" for s in SUBJECTS))
    for spec in PROBE_SPECS:
        vals = [res[s]["cv_retrieval"]["per_channel"][spec]["top5"] for s in SUBJECTS]
        print(f"  {spec:44} " + "".join(f"{v*100:>9.2f}%" for v in vals))

    print("\n  alpha-band (8-13 Hz) power fraction:")
    print(f"  {'measure':20} " + "".join(f"{s:>10}" for s in SUBJECTS))
    for key, lbl in (("alpha_frac_mean", "alpha mean"),
                     ("alpha_frac_max", "alpha max ch"),
                     ("alpha_frac_min", "alpha min ch")):
        vals = [res[s]["spectrum"][key] for s in SUBJECTS]
        print(f"  {lbl:20} " + "".join(f"{v:>10.4f}" for v in vals))
    print("=" * 86)
    print(f"\n  wrote {(OUT / 'exp4_sub03_regression.json').relative_to(RESULTS_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
