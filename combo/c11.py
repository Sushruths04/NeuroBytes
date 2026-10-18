"""C11 = C10 + two contrastive neural families (conv and linear trunk, combo/neural_clip.py).
score = (1 - a - b) * C10_blend + a * rrf+hub(NC_conv) + b * rrf+hub(NC_lin), then CSLS k=2, Sinkhorn(3, 0.5).
a, b chosen on validation (combo/exp_nc2.py). NC test matrices: outputs/<NC_DIR>/<sub>__<TAG>_all.pkl (dev, from the per-subject
VAL runs: trained on all training rows minus that subject's VAL videos) and <sub>__<TAG>_final_all.pkl (all training rows).
    python combo/c11.py --dev            # dev test sub-01..04
    python combo/c11.py --out <dir>      # sub-05/06 ranked lists
"""
import argparse, hashlib, json, os, pickle, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c10  # noqa: E402

NC_DIR = os.environ.get('NC_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c11_inputs'))
FAMILIES = [(f.split(':')[0], float(f.split(':')[1])) for f in os.environ.get('NC_FAMILIES', 'C11conv:0.1').split(',')]
FINAL_FOR_DEV = os.environ.get('NC_FINAL_FOR_DEV', '1') == '1'   # dev scores from the all-training model (as for sub-05/06)
# The shipped component scores (winfuse, ours77, T2) follow the row order of the official tst files. With C13_ALIGN_TO_DATA=1
# (set by the quickstart and Docker, whose network scores come from the mounted EEG) each mounted test recording is matched to
# its official row by a fingerprint of its values, so a reordered tst file gives the same per-recording rankings.
ALIGN = os.environ.get('C13_ALIGN_TO_DATA', '0') == '1'
FINGERPRINTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c10_inputs', 'tst_row_fingerprints.json')


def order(sub):
    """Index p with mounted_tst[i] == official_tst[p[i]]; identity unless C13_ALIGN_TO_DATA=1."""
    if not ALIGN:
        return np.arange(102)
    X = np.load(os.path.join(os.environ['WINFUSE_DATA_ROOT'], 'tst', f'{sub}_tst.npy'))
    ref = {h: i for i, h in enumerate(json.load(open(FINGERPRINTS))[sub])}
    fp = [hashlib.sha256(np.ascontiguousarray(x, dtype='<f8').tobytes()).hexdigest()[:20] for x in X]
    missing = [i for i, h in enumerate(fp) if h not in ref]
    if missing or len(fp) != 102:
        sys.exit(f'{sub}: {len(missing)} test recordings are not in the official tst file. The shipped component scores only '
                 f'cover the official recordings; rebuild all components from raw EEG with scripts/run_c13_pipeline.sh.')
    return np.array([ref[h] for h in fp])


def blend(sub, t2_test):
    p = order(sub)
    W = np.load(f'{c10.WF}/{sub}_fused_scores.npy').astype(float)[p]; O = np.load(f'{c10.O77}/{sub}_ours77.npy').astype(float)[p]
    t2_test = np.asarray(t2_test)[p]
    n = c10.N_WF + c10.N_OURS
    return (1 - c10.ALPHA) * ((c10.N_WF / n) * W + (c10.N_OURS / n) * O) + c10.ALPHA * c10.rrf_hub(t2_test)


def nc(sub, tag):
    f = f'{NC_DIR}/{sub}__{tag}_final_all.pkl'
    if sub in ('sub-05', 'sub-06') or (FINAL_FOR_DEV and os.path.exists(f)):
        return pickle.load(open(f, 'rb'))['test']
    return pickle.load(open(f'{NC_DIR}/{sub}__{tag}_all.pkl', 'rb'))['test']


def c11_scores(sub, t2_test, use_sinkhorn=True):
    w = sum(b for _, b in FAMILIES)
    S = (1 - w) * blend(sub, t2_test) + sum(b * c10.rrf_hub(nc(sub, tag)) for tag, b in FAMILIES)
    S = c10.csls(S, c10.K)
    return c10.sinkhorn(S) if use_sinkhorn else S


def dev(use_sinkhorn=True):
    R = []
    for sub in ('sub-01', 'sub-02', 'sub-03', 'sub-04'):
        t2 = pickle.load(open(f'{c10.T2}/{sub}__T2_mindread_enc.pkl', 'rb'))
        r = c10.ranks(c11_scores(sub, t2['test'], use_sinkhorn), np.asarray(t2['truth'])[order(sub)]); R.append(r)
        print(f'{sub} top1 {(r == 1).mean() * 100:5.1f} top5 {(r <= 5).mean() * 100:5.1f} rank {r.mean():5.2f}')
    r = np.concatenate(R)
    print(f'DEV-TEST sub-01..04 top1 {(r == 1).mean() * 100:5.2f} top5 {(r <= 5).mean() * 100:5.2f} rank {r.mean():5.2f}')
    return R


def submit(out, use_sinkhorn=True):
    os.makedirs(out, exist_ok=True)
    for sub in ('sub-05', 'sub-06'):
        S = c11_scores(sub, np.load(f'{c10.T2}/T2_test_{sub}.npy'), use_sinkhorn)
        r = c10.TEST_IDS[np.argsort(-S, 1, kind='stable')].astype(np.int64)
        assert r.shape == (102, 102) and all(set(row) == set(c10.TEST_IDS.tolist()) for row in r)
        np.save(f'{out}/{sub}_ranked_candidate_list.npy', r); print('wrote', f'{out}/{sub}_ranked_candidate_list.npy')


def check(out):
    """Compare regenerated ranked lists in <out> with submissions/C13 (matched per recording when the tst rows are reordered)."""
    ref_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'submissions', 'C13')
    for f in sorted(x for x in os.listdir(out) if x.endswith('_ranked_candidate_list.npy')):
        a = np.load(os.path.join(out, f)); b = np.load(os.path.join(ref_dir, f))[order(f.split('_')[0])]
        assert a.shape == (102, 102) and a.dtype.kind == 'i' and all(set(r) == set(c10.TEST_IDS.tolist()) for r in a), f'{f}: format check failed'
        t1 = np.mean(a[:, 0] == b[:, 0]) * 100; t5 = np.mean([set(x[:5]) == set(y[:5]) for x, y in zip(a, b)]) * 100
        note = 'byte-identical' if (a == b).all() else 'deep ranks differ slightly (CPU/GPU rounding)'
        print(f'{f}: format OK; top-1 {t1:.0f}% and top-5 {t5:.0f}% identical to submissions/C13; {note}')
        assert t1 == 100 and t5 == 100, f'{f}: top-1/top-5 differ from submissions/C13'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--dev', action='store_true'); ap.add_argument('--out'); ap.add_argument('--no-sinkhorn', action='store_true')
    ap.add_argument('--check', help='folder with regenerated ranked lists to compare with submissions/C13')
    a = ap.parse_args()
    if a.dev:
        dev(not a.no_sinkhorn)
    if a.out:
        submit(a.out, not a.no_sinkhorn)
    if a.check:
        check(a.check)
