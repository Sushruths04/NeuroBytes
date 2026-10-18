"""C10: final assembly of the combined submission. Inputs: combo/c10_inputs (ours77 + T2 scores, shipped) and
the aligned winfuse scores (results/winfuse/scores, or set WF_SCORES).

score = 0.85 * ((509/586) * WF_aligned + (77/586) * OURS77) + 0.15 * rrf+hub(T2)
      -> CSLS k=2 over the subject's 102 test recordings
      -> Sinkhorn (3 iterations, tau 0.5) over the 102 x 102 matrix

WF_aligned = cross-subject-aligned winfuse fused scores (509 equal-weight scorers, results/winfuse/scores).
OURS77     = equal-weight rrf+hub mean of our 77 feature scorers (modal_combo.py --o77).
T2         = mindread T2.swap encoder scores (t2_scores.py).
alpha 0.15 and k 2 were chosen on winfuse VALIDATION draws (C3/C5); Sinkhorn (3, 0.5) is the winfuse setting, chosen on
validation over 5 splits. 509/586 vs 77/586 is the equal per-scorer weight of the C5 selection. Settings were informed by training-video validation; development labels were consulted
during the broader model development. Sinkhorn is a transductive post-process over all test recordings (like Hungarian); the
fallback without it is written with --no-sinkhorn.

  python combo/c10.py --dev                      # dev test sub-01..04 (logged check)
  python combo/c10.py --out <dir> [--no-sinkhorn] # sub-05/06 ranked lists
"""
import argparse, os, pickle, sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = os.path.join(REPO, 'combo', 'c10_inputs')
WF = os.environ.get('WF_SCORES', os.path.join(REPO, 'results/winfuse/scores'))
O77 = os.environ.get('O77_DIR', INPUTS)
T2 = os.environ.get('T2_DIR', INPUTS)
sys.path.insert(0, os.path.join(REPO, 'src'))
from mindread.methods import csls  # noqa: E402

TEST_IDS = np.arange(1001, 1103)
ALPHA, K, SK_ITERS, SK_TAU, N_WF, N_OURS = 0.15, 2, 3, 0.5, 509, 77


def rrf_hub(S, k=60):
    """winfuse 'rrf+hub' transform: reciprocal-rank fusion score per row, then column (hub) centring."""
    S = np.asarray(S, dtype=np.float64)
    r = np.empty_like(S)
    r[np.arange(S.shape[0])[:, None], np.argsort(-S, 1)] = np.arange(1, S.shape[1] + 1)
    R = 1.0 / (k + r)
    return R - R.mean(0, keepdims=True)


def sinkhorn(S, iters=SK_ITERS, tau=SK_TAU):
    """winfuse Sinkhorn (results/winfuse/rank.py): row z-score, exp(z/tau), alternate row/column normalisation."""
    S = np.asarray(S, dtype=np.float64)
    Z = (S - S.mean(1, keepdims=True)) / np.maximum(S.std(1, keepdims=True), 1e-12)
    M = np.exp(np.clip(Z / tau, -700, 700))
    for _ in range(iters):
        M = M / np.maximum(M.sum(1, keepdims=True), 1e-300)
        M = M / np.maximum(M.sum(0, keepdims=True), 1e-300)
    return M


def c10_scores(sub, t2_test, use_sinkhorn=True):
    W = np.load(f'{WF}/{sub}_fused_scores.npy').astype(float)
    O = np.load(f'{O77}/{sub}_ours77.npy').astype(float)
    n = N_WF + N_OURS
    S = csls((1 - ALPHA) * ((N_WF / n) * W + (N_OURS / n) * O) + ALPHA * rrf_hub(t2_test), K)
    return sinkhorn(S) if use_sinkhorn else S


def ranks(S, truth):
    return (np.argsort(-S, 1, kind='stable') == truth[:, None]).argmax(1) + 1


def dev(use_sinkhorn):
    allr = []
    for sub in ('sub-01', 'sub-02', 'sub-03', 'sub-04'):
        t2 = pickle.load(open(f'{T2}/{sub}__T2_mindread_enc.pkl', 'rb'))
        r = ranks(c10_scores(sub, t2['test'], use_sinkhorn), np.asarray(t2['truth']))
        allr.append(r)
        print(f'{sub} top1 {(r == 1).mean() * 100:5.1f} top5 {(r <= 5).mean() * 100:5.1f} rank {r.mean():5.2f}')
    r = np.concatenate(allr)
    print(f'DEV-TEST sub-01..04 top1 {(r == 1).mean() * 100:5.2f} top5 {(r <= 5).mean() * 100:5.2f} rank {r.mean():5.2f}')


def submit(out, use_sinkhorn):
    os.makedirs(out, exist_ok=True)
    for sub in ('sub-05', 'sub-06'):
        S = c10_scores(sub, np.load(f'{T2}/T2_test_{sub}.npy'), use_sinkhorn)
        r = TEST_IDS[np.argsort(-S, 1, kind='stable')].astype(np.int64)
        assert r.shape == (102, 102) and all(set(row) == set(TEST_IDS.tolist()) for row in r)
        np.save(f'{out}/{sub}_ranked_candidate_list.npy', r)
        print('wrote', f'{out}/{sub}_ranked_candidate_list.npy')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev', action='store_true')
    ap.add_argument('--out')
    ap.add_argument('--no-sinkhorn', action='store_true')
    a = ap.parse_args()
    if a.dev:
        dev(not a.no_sinkhorn)
    if a.out:
        submit(a.out, not a.no_sinkhorn)
