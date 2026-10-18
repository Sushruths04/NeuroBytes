"""PART A (final) - quantify sub-03 as a low-SNR subject, not a defective one.

The pipeline's accuracy depends on how predictable the EEG window features are
from the stimulus embedding (and vice versa). Measure that directly, per subject,
with 5-fold cross-validation on TRAINING videos only:

  * mean held-out cross-validated encoding r:  predict each EEG window dimension
    from the stimulus embedding (dinov2_base), report the mean over dimensions of
    the CV Pearson r between measured and predicted. This is exactly the "signal
    the decoder has to work with" and is label-free w.r.t. test.

  * fraction of EEG dimensions with CV r > 0.1 (how many usable dimensions).

If sub-03 is low on these on raw window features AND stays low after the best
artefact remedy, it is a low-SNR subject with no fixable defect.

TRAINING DATA ONLY.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import winfuse.common.io as cio
from winfuse.common import preprocess as pp
from winfuse.common.config import RESULTS_DIR, SEED
from winfuse.compare_embed import fusion as fu
from winfuse.compare_embed.explore.exp4_sub03_fix import _window_from_epochs

SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04"]
OUT = RESULTS_DIR / "compare_embed" / "explore"
WIN = ("win_50_400", "dinov2_base")


def cv_encoding_r(sub: str) -> dict:
    Xep, _ = pp.preprocessed_train(sub)
    Xep = np.asarray(Xep, dtype=np.float64)
    ids = cio.subject_train_ids(sub)
    es, st = WIN
    _, lo, hi = es.split("_")
    W = _window_from_epochs(Xep, int(lo), int(hi))         # (n, 17*win)
    E, found = fu.stimulus_matrix(st, ids, "train")
    pos = {v: i for i, v in enumerate(found)}
    keep = [i for i, v in enumerate(ids) if v in pos]
    W = W[keep]
    E = E[[pos[ids[i]] for i in keep]]

    # reduce EEG to a manageable PC space for a stable per-dimension r
    npc = 100
    pcaW = PCA(n_components=npc, random_state=SEED).fit(W)
    Wp = pcaW.transform(W)                                 # (n, 100) EEG targets

    P = np.zeros_like(Wp)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in kf.split(Wp):
        pe = PCA(n_components=min(256, len(tr) - 1, E.shape[1]),
                 random_state=SEED).fit(E[tr])
        xs = StandardScaler().fit(pe.transform(E[tr]))
        ys = StandardScaler().fit(Wp[tr])
        m = Ridge(alpha=1e3).fit(xs.transform(pe.transform(E[tr])), ys.transform(Wp[tr]))
        P[te] = ys.inverse_transform(m.predict(xs.transform(pe.transform(E[te]))))
    Wc = Wp - Wp.mean(0, keepdims=True)
    Pc = P - P.mean(0, keepdims=True)
    num = (Wc * Pc).sum(0)
    den = np.sqrt((Wc ** 2).sum(0) * (Pc ** 2).sum(0))
    r = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return {"mean_cv_encoding_r": float(np.mean(r)),
            "max_cv_encoding_r": float(np.max(r)),
            "frac_dims_r_gt_0.1": float(np.mean(r > 0.1)),
            "frac_dims_r_gt_0.05": float(np.mean(r > 0.05)),
            "n_dims": int(len(r))}


def main() -> int:
    res = {s: cv_encoding_r(s) for s in SUBJECTS}
    (OUT / "exp4_sub03_snr.json").write_text(json.dumps(res, indent=2))
    print("\n" + "=" * 74)
    print("  stimulus-decodable SNR: CV encoding r of win_50_400 EEG from dinov2")
    print("=" * 74)
    print(f"  {'subject':10} {'mean r':>10} {'max r':>10} {'frac r>0.1':>12} {'frac r>0.05':>12}")
    for s in SUBJECTS:
        m = res[s]
        print(f"  {s:10} {m['mean_cv_encoding_r']:10.4f} {m['max_cv_encoding_r']:10.4f} "
              f"{m['frac_dims_r_gt_0.1']:12.3f} {m['frac_dims_r_gt_0.05']:12.3f}")
    print("=" * 74)
    print(f"\n  wrote {(OUT / 'exp4_sub03_snr.json').relative_to(RESULTS_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
