"""Score raw test EEG with saved C13 network weights (no feature banks or training data needed).

    python combo/nc_predict.py --weights weights/c13 --tag h6 --subject sub-05 --eeg <sub-05_tst.npy> --out <dir>
Inputs: <weights>/<tag>_seed*.pt (state dicts + target means), <tag>_candidates.pt (candidate embeddings for video ids
1-1102 and the training config), <weights>/preproc/<sub>_prep.joblib (that person's train-fitted preprocessor).
Writes <out>/<sub>__<tag>_final_all.pkl {'test': 102 x 102 scores}, the same format combo/c11.py reads.
"""
import argparse, glob, os, pickle, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--subject', required=True); ap.add_argument('--eeg', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--candidates', default='1001-1102', help='candidate video id range, e.g. 1001-1102')
    a = ap.parse_args()
    import torch
    C = torch.load(os.path.join(a.weights, f'{a.tag}_candidates.pt'), weights_only=False, map_location='cpu')
    cfg = C['config']
    os.environ.update(NC_T=str(cfg['T']), NC_HID=str(cfg['HID']), NC_ARCH=cfg['ARCH'], NC_ADAPT='1' if cfg['ADAPT'] else '0',
                      NC_TARGETS=','.join(C['targets']))
    sys.path.insert(0, HERE)
    import neural_clip as nc                                   # same model definition as training
    import joblib
    lo, hi = map(int, a.candidates.split('-')); cid = list(range(lo, hi + 1)); pos = {v: i for i, v in enumerate(C['ids'])}
    prep = joblib.load(os.path.join(a.weights, 'preproc', f'{a.subject}_prep.joblib'))
    X = nc.prep(prep.transform(np.load(a.eeg).astype(np.float32)))
    k = C['subjects'].index(a.subject)
    acc = {t: 0 for t in C['targets']}
    seeds = sorted(glob.glob(os.path.join(a.weights, f'{a.tag}_seed*.pt')))
    if len(seeds) != 5:
        raise ValueError(f'Expected five checkpoints for {a.tag}, found {len(seeds)}')
    for f in seeds:
        W = torch.load(f, weights_only=False, map_location='cpu')
        net = nc.Net(W['dims'], hid=cfg['HID']).to(nc.DEVICE)
        net(torch.tensor(X[:2]).to(nc.DEVICE), torch.zeros(2, dtype=torch.long, device=nc.DEVICE))   # materialise lazy layers
        net.load_state_dict(W['state_dict']); net.eval()
        cand = {t: C['cand'][t].float().numpy()[[pos[v] for v in cid]] for t in C['targets']}
        sc = nc.score(net, W['mu'], X, k, cand)
        for t in C['targets']:
            acc[t] = acc[t] + sc[t] / len(seeds)
    S = sum(acc.values()) / len(acc)
    os.makedirs(a.out, exist_ok=True)
    pickle.dump({'test': S}, open(os.path.join(a.out, f'{a.subject}__{a.tag}_final_all.pkl'), 'wb'))
    print('scored', a.subject, a.tag, S.shape, 'seeds', len(seeds))


if __name__ == '__main__':
    main()
