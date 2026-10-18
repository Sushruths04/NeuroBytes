"""COMBO = winfuse (code imported unchanged) + our stimulus features as extra scorers.

New scorers: each of winfuse's 15 latency windows x our features (V-JEPA 2 giant L30, V-JEPA 2 large, S3D, CLAP audio,
SigLIP), decoding direction, plus encoding on win_50_400 for V-JEPA-G and S3D. They are built exactly like winfuse's
own channels (same preprocessing, same TRAIN/VAL split, same Channel class) and stored in its channel cache, so
winfuse's own --select picks channel set, transform and weights on VALIDATION over 509 + new scorers.

    python combo.py build sub-01         # new scorers for one development subject (val + dev-test matrices)
    python combo.py select               # winfuse select over all scorers + winfuse dev test
    python combo.py predict OUT sub-05   # ranked list for one subject (fit on its training recordings)
"""
import importlib, importlib.abc, importlib.util, json, os, pickle, sys, time

import numpy as np

_TOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_TOP, 'src', 'winfuse')):    # inside the repo: <repo>/combo/; data and outputs next to the repo
    SRC, MREPO, ND = [os.path.join(_TOP, 'src')], _TOP, _TOP
else:                                                       # development workspace layout
    SRC, MREPO, ND = [os.path.join(_TOP, 'winfuse_pr13', 'src'), os.path.join(_TOP, 'mindread', 'src')], os.path.join(_TOP, 'mindread'), _TOP
DATA = os.environ.get('WINFUSE_DATA_ROOT', os.path.join(ND, 'data', 'challenge_official', 'eeg-mind-reading-challenge'))
os.environ.setdefault('WINFUSE_DATA_ROOT', DATA)
for _p in SRC:
    sys.path.insert(0, _p)
OURS = os.environ.get('OURS', os.path.join(ND, 'outputs', 'mindread_features'))


class _SrcAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """winfuse caches were pickled when the package was imported as src.*"""
    def find_spec(self, name, path=None, target=None):
        if name == 'src' or name.startswith('src.'):
            return importlib.util.spec_from_loader(name, self, is_package=True)

    def create_module(self, spec):
        return importlib.import_module('winfuse' + spec.name[3:])

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _SrcAlias())
from winfuse.common import io as cio, preprocess as pp  # noqa: E402
from winfuse.common.config import SFREQ  # noqa: E402
from winfuse.common.feature_api import get_extractor  # noqa: E402
from winfuse.compare_embed import eeg_features as ef, fusion as fu, pipeline as P  # noqa: E402

if os.environ.get('COMBO_CACHE'):                          # dry runs write channel pickles elsewhere
    from pathlib import Path
    P.CACHE = Path(os.environ['COMBO_CACHE']); P.CACHE.mkdir(parents=True, exist_ok=True)
OUR_STIMS = ['vjepa2_g_l30', 'vjepa2_l_l12_18_24', 's3d_k400', 'clap_audio', 'siglip_b16']
WINDOWS = [f'win_{a}_{b}' for a, b in P.WINDOWS_MS]
NEW_SPECS = ([f'{w}:ours_{s}:dec' for w in WINDOWS for s in OUR_STIMS]
             + [f'win_50_400:ours_{s}:enc' for s in ('vjepa2_g_l30', 's3d_k400')])
T2_SPEC = 'T2:mindread:enc'                               # our frozen T2 as one scorer (combo/t2_scores.py)
T2_DIR = os.environ.get('T2_DIR', os.path.join(ND, 'outputs', 'combo_t2_scores_20261017'))
REF_SPEC = 'win_50_400:resnet50@layer_layer3:dec'      # an existing winfuse channel, to check split alignment
_tables = {}


def ours(name, ids):
    if name not in _tables:
        z = np.load(os.path.join(OURS, f'{name}.npz'))
        _tables[name] = {int(i): v for i, v in zip(z['ids'], z['feats'] if 'feats' in z.files else z[z.files[-1]])}
    t = _tables[name]
    return np.stack([np.asarray(t[int(i)], dtype=np.float64).ravel() for i in ids])


def stim(st, ids, split):
    if st.startswith('ours_'):
        return ours(st[5:], ids)
    X, found = fu.stimulus_matrix(st, list(ids), split)
    assert list(found) == list(ids), f'{st}: missing ids'
    return X


def window(X, es):
    _, lo, hi = es.split('_')
    i0, i1 = max(int(round(int(lo) / 1000 * SFREQ)), 0), min(int(round(int(hi) / 1000 * SFREQ)), X.shape[2])
    return np.asarray(X[:, :, i0:i1], dtype=np.float64).reshape(len(X), -1)


def build(sub, specs=NEW_SPECS):
    t0 = time.time()
    Xtr, ids = pp.preprocessed_train(sub)
    ids = list(ids)
    itr, iva = P.split_ids(ids)
    ref = P.cpath(sub, REF_SPEC)
    if ref.exists():                                       # same TRAIN/VAL rows as winfuse's cached channels
        r = pickle.load(open(ref, 'rb'))
        assert (r['n_val'], r['n_train']) == (len(iva), len(itr)), (r['n_val'], r['n_train'], len(iva), len(itr))
    Xte = pp.preprocessed_test(sub, reason='combo build (dev subject)')
    truth = np.array([P.TEST_IDS.index(v) for v in cio.load_test_labels(sub, reason='combo build')])
    for k, spec in enumerate(specs):
        es, st, _ = spec.split(':')
        F, Fte = window(Xtr, es), window(Xte, es)
        E, Ec = stim(st, ids, 'train'), stim(st, P.TEST_IDS, 'test')
        val = fu.Channel(spec).fit(F[itr], E[itr]).score(F[iva], E[iva]).astype(np.float32)
        test = fu.Channel(spec).fit(F, E).score(Fte, Ec).astype(np.float32)
        with open(P.cpath(sub, spec), 'wb') as f:
            pickle.dump({'val': val, 'test': test, 'truth': truth, 'spec': spec,
                         'n_val': len(iva), 'n_train': len(itr)}, f)
        if k % 20 == 0:
            print(sub, k, '/', len(specs), round(time.time() - t0), 's', flush=True)
    print('built', sub, len(specs), round(time.time() - t0), 's', flush=True)


def select_and_test():
    subs = list(P.DEV_SUBJECTS)
    specs = P.channel_list() + NEW_SPECS + [T2_SPEC]
    try:
        fz = P.select(subs, specs)
    except ValueError as e:                                # winfuse prints paths relative to ROOT; harmless
        print('select print error (ignored):', str(e)[:80])
        fz = json.loads(P.FROZEN.read_text())
    print('T2 in selection:', T2_SPEC in fz['channels'],
          'weight %.3f' % sum(w for s, w in zip(fz['channels'], fz['weights']) if s == T2_SPEC))
    n_new = sum(s in NEW_SPECS for s in fz['channels'])
    w_new = sum(w for s, w in zip(fz['channels'], fz['weights']) if s in NEW_SPECS)
    print('COMBO selected', fz['selected'], '| channels', len(fz['channels']), '| ours', n_new,
          '| weight on ours %.3f' % w_new, '| VAL top-1 %.2f top-5 %.2f' % (fz['val']['top1'] * 100, fz['val']['top5'] * 100))
    try:
        P.test(subs)
    except ValueError as e:
        print('test print error (ignored):', str(e)[:80])
    return fz


def predict(out, sub, s, frozen=None):
    """Ranked list for subject number s using the frozen selection (fit on its training recordings)."""
    from mindread.data import load_test
    from mindread.scoring import to_ranked_ids, validate_submission
    fz = json.load(open(frozen)) if frozen else json.loads(P.FROZEN.read_text())
    specs, tf, w = fz['channels'], P.TRANSFORMS[fz['transform']], np.array(fz['weights'])
    t0 = time.time()
    Xtr, ids = pp.preprocessed_train(sub)
    ids = list(ids)
    Xte = pp.get_preprocessor(sub).transform(load_test(DATA, s, final_inference=s in (5, 6)))
    feats = {}
    M = np.zeros((len(Xte), len(P.TEST_IDS)))
    for k, spec in enumerate(specs):
        es, st, _ = spec.split(':')
        if spec == T2_SPEC:
            M += w[k] * tf(np.load(os.path.join(T2_DIR, f'T2_test_{sub}.npy')).astype(np.float64))
            continue
        if es not in feats:
            if es.startswith('win_'):
                feats[es] = (window(Xtr, es), ids, window(Xte, es))
            else:
                Ftr, fids = ef.eeg_feature_matrix(sub, es)
                x = get_extractor(es); x.fit(Xtr)
                feats[es] = (np.asarray(Ftr, dtype=np.float64), list(fids), np.asarray(x.transform(Xte), dtype=np.float64))
        Ftr, fids, Fte = feats[es]
        if st.startswith('ours_'):
            E, keep = ours(st[5:], fids), list(range(len(fids)))
        else:
            E, found = fu.stimulus_matrix(st, list(fids), 'train')
            pos = {v: i for i, v in enumerate(found)}
            keep = [i for i, v in enumerate(fids) if v in pos]
            E = E[[pos[fids[i]] for i in keep]]
        c = fu.Channel(spec).fit(Ftr[keep], E)
        M += w[k] * tf(c.score(Fte, stim(st, P.TEST_IDS, 'test')))
    ranked = to_ranked_ids(M, np.array(P.TEST_IDS)); validate_submission(ranked)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, f'{sub}_combo_scores.npy'), M)
    np.save(os.path.join(out, f'{sub}_ranked_candidate_list.npy'), ranked)
    print('predicted', sub, len(specs), 'channels', round(time.time() - t0), 's', flush=True)


def blend_select(alphas=tuple(np.round(np.arange(0, 0.95, 0.05), 2))):
    """alpha for (1-a)*combo + a*T2 chosen on winfuse VAL pools (dev subjects), then reported on the dev test."""
    fz = json.loads(P.FROZEN.read_text())
    specs = [s for s in fz['channels'] if s != T2_SPEC]
    tf = P.TRANSFORMS[fz['transform']]
    w = np.array([wt for s, wt in zip(fz['channels'], fz['weights']) if s != T2_SPEC]); w = w / w.sum()
    res = {}
    for sub in P.DEV_SUBJECTS:
        per = P.load(sub, specs + [T2_SPEC])
        mats = {}
        for part in ('val', 'test'):
            mats[part] = (sum(w[i] * tf(per[s][part]) for i, s in enumerate(specs)), tf(per[T2_SPEC][part]))
        res[sub] = (mats, per[T2_SPEC]['truth'], P.make_pools(per[T2_SPEC]['val'].shape[0]))
    table = []
    for a in alphas:
        vm, tm = [], []
        for sub, (mats, truth, pools) in res.items():
            C, T = mats['val']; vm.append(P.metrics_from_pool_ranks(P.pool_ranks((1 - a) * C + a * T, pools)))
            C, T = mats['test']; r = fu.ranks_of_truth((1 - a) * C + a * T, truth); tm.append(r)
        v = {k: float(np.mean([m[k] for m in vm])) for k in vm[0]}; r = np.concatenate(tm)
        table.append((a, v['top1'], v['top5'], v['mrr'], float((r == 1).mean()), float((r <= 5).mean())))
        print('alpha %.2f | VAL top1 %.2f top5 %.2f mrr %.4f | DEV-TEST top1 %.2f top5 %.2f'
              % (a, v['top1'] * 100, v['top5'] * 100, v['mrr'], table[-1][4] * 100, table[-1][5] * 100), flush=True)
    best = max(table, key=lambda t: t[2])                   # chosen on VAL top-5 only
    print('BLEND chosen on VAL: alpha %.2f -> DEV-TEST top1 %.2f top5 %.2f' % (best[0], best[4] * 100, best[5] * 100))
    return float(best[0])


def family(spec):
    if spec == T2_SPEC:
        return 'T2'
    es, st, d = spec.split(':')
    return st.split('@')[0] + ('_enc' if d == 'enc' else '')


def family_fit(l2=1.0, n_draws=20):
    """Per-family weights on top of the equal-weight combo anchor, fitted with mindread's ResidualFusion on
    102-way draws from winfuse VAL (dev subjects pooled); reported on the dev test. Returns {family: weight}."""
    from mindread.eval import retrieval_draws
    from mindread.fusion import ResidualFusion, row_z
    fz = json.loads(P.FROZEN.read_text())
    tf = P.TRANSFORMS[fz['transform']]
    specs = [s for s in fz['channels'] if s != T2_SPEC]
    fams = sorted({family(s) for s in specs}) + ['T2']
    cal, truth, test = {n: [] for n in ['baseline'] + fams}, [], {}
    for k, sub in enumerate(P.DEV_SUBJECTS):
        per = P.load(sub, specs + [T2_SPEC])
        mats = {}
        for part in ('val', 'test'):
            T = {s: tf(per[s][part]) for s in specs + [T2_SPEC]}
            m = {'baseline': np.mean([T[s] for s in specs], 0)}
            for f in fams:
                m[f] = np.mean([T[s] for s in specs + [T2_SPEC] if family(s) == f], 0)
            mats[part] = m
        nv = per[T2_SPEC]['val'].shape[0]
        for c in retrieval_draws(np.arange(nv), 102, n_draws, 17 + k):
            for n in cal:
                cal[n].append(mats['val'][n][np.ix_(c, c)])
            truth.append(np.arange(102))
        test[sub] = (mats['test'], per[T2_SPEC]['truth'])
    fus = ResidualFusion(fams, l2=l2).fit({n: np.concatenate(v) for n, v in cal.items()}, np.concatenate(truth))
    w = dict(zip(fams, np.round(fus.weights, 4).tolist()))
    print('FAMILY weights:', {f: x for f, x in w.items() if x > 0.001}, flush=True)
    for name, scorer in (('equal combo', lambda m: m['baseline']), ('family-weighted', lambda m: fus.scores(m))):
        r = np.concatenate([fu.ranks_of_truth(scorer(m), t) for m, t in test.values()])
        print('DEV-TEST %-16s top1 %.2f top5 %.2f mean rank %.2f' % (name, (r == 1).mean() * 100, (r <= 5).mean() * 100, r.mean()), flush=True)
    return w


def predict_families(out, sub, s, w):
    """sub-05/06 scores with family weights: refits every channel on the subject's training recordings."""
    from mindread.data import load_test
    from mindread.fusion import row_z
    from mindread.scoring import to_ranked_ids, validate_submission
    fz = json.loads(P.FROZEN.read_text()); tf = P.TRANSFORMS[fz['transform']]
    Xtr, ids = pp.preprocessed_train(sub); ids = list(ids)
    Xte = pp.get_preprocessor(sub).transform(load_test(DATA, s, final_inference=s in (5, 6)))
    feats, sums, cnt = {}, {}, {}
    for spec in fz['channels']:
        f = family(spec)
        if spec == T2_SPEC:
            S = np.load(os.path.join(T2_DIR, f'T2_test_{sub}.npy')).astype(np.float64)
        else:
            es, st, _ = spec.split(':')
            if es not in feats:
                if es.startswith('win_'):
                    feats[es] = (window(Xtr, es), ids, window(Xte, es))
                else:
                    Ftr, fids = ef.eeg_feature_matrix(sub, es)
                    x = get_extractor(es); x.fit(Xtr)
                    feats[es] = (np.asarray(Ftr, dtype=np.float64), list(fids), np.asarray(x.transform(Xte), dtype=np.float64))
            Ftr, fids, Fte = feats[es]
            if st.startswith('ours_'):
                E, keep = ours(st[5:], fids), list(range(len(fids)))
            else:
                E, found = fu.stimulus_matrix(st, list(fids), 'train')
                pos = {v: i for i, v in enumerate(found)}
                keep = [i for i, v in enumerate(fids) if v in pos]
                E = E[[pos[fids[i]] for i in keep]]
            S = fu.Channel(spec).fit(Ftr[keep], E).score(Fte, stim(st, P.TEST_IDS, 'test'))
        T = tf(S)
        if spec != T2_SPEC:
            sums['baseline'] = sums.get('baseline', 0) + T; cnt['baseline'] = cnt.get('baseline', 0) + 1
        sums[f] = sums.get(f, 0) + T; cnt[f] = cnt.get(f, 0) + 1
    M = row_z(sums['baseline'] / cnt['baseline']) + sum(w[f] * row_z(sums[f] / cnt[f]) for f in w if f in sums and w[f] > 0)
    r = to_ranked_ids(M, np.array(P.TEST_IDS)); validate_submission(r)
    os.makedirs(out, exist_ok=True); np.save(os.path.join(out, f'{sub}_ranked_candidate_list.npy'), r)
    print('family predict', sub, flush=True)


def csls_select(alpha=0.15, ks=(0, 2, 3, 5, 10, 20), n_draws=20):
    """CSLS strength k on the final C3 blend, chosen on winfuse VAL 102-way draws (dev subjects); dev test reported."""
    from mindread.eval import retrieval_draws
    from mindread.methods import csls
    fz = json.loads(P.FROZEN.read_text()); tf = P.TRANSFORMS[fz['transform']]
    specs = [s for s in fz['channels'] if s != T2_SPEC]
    w = np.array([wt for s, wt in zip(fz['channels'], fz['weights']) if s != T2_SPEC]); w = w / w.sum()
    data = []
    for k, sub in enumerate(P.DEV_SUBJECTS):
        per = P.load(sub, specs + [T2_SPEC])
        B = {part: (1 - alpha) * sum(w[i] * tf(per[s][part]) for i, s in enumerate(specs)) + alpha * tf(per[T2_SPEC][part])
             for part in ('val', 'test')}
        data.append((B, per[T2_SPEC]['truth'], list(retrieval_draws(np.arange(B['val'].shape[0]), 102, n_draws, 17 + k))))
    rows = []
    for kk in ks:
        f = (lambda S: S) if kk == 0 else (lambda S, kk=kk: csls(S, kk))
        rv = np.concatenate([fu.ranks_of_truth(f(B['val'][np.ix_(c, c)]), np.arange(102)) for B, _, dr in data for c in dr])
        rt = np.concatenate([fu.ranks_of_truth(f(B['test']), t) for B, t, _ in data])
        rows.append((kk, (rv == 1).mean(), (rv <= 5).mean(), (rt == 1).mean(), (rt <= 5).mean()))
        print('CSLS k=%2d | VAL top1 %.2f top5 %.2f | DEV-TEST top1 %.2f top5 %.2f' % (kk, rows[-1][1] * 100, rows[-1][2] * 100,
              rows[-1][3] * 100, rows[-1][4] * 100), flush=True)
    best = max(rows, key=lambda r: (r[1], r[2]))              # VAL top-1, then VAL top-5
    print('CSLS chosen on VAL: k=%d -> DEV-TEST top1 %.2f top5 %.2f' % (best[0], best[3] * 100, best[4] * 100), flush=True)
    return int(best[0])


def blend_submit(out, alpha, subs=('sub-05', 'sub-06'), csls_k=0):
    """Final lists: (1-alpha)*combo scores (from predict) + alpha*rrf+hub(T2 test scores, combo/t2_scores.py),
    then CSLS(k) over the subject's test recordings if csls_k > 0 (C5: alpha 0.15, k 2, both chosen on VAL)."""
    from mindread.methods import csls
    from mindread.scoring import to_ranked_ids, validate_submission
    tf = P.TRANSFORMS['rrf+hub']
    for sub in subs:
        S = (1 - alpha) * np.load(os.path.join(out, f'{sub}_combo_scores.npy')) + \
            alpha * tf(np.load(os.path.join(T2_DIR, f'T2_test_{sub}.npy')).astype(np.float64))
        if csls_k:
            S = csls(S, csls_k)
        r = to_ranked_ids(S, np.array(P.TEST_IDS)); validate_submission(r)
        np.save(os.path.join(out, f'{sub}_ranked_candidate_list.npy'), r)
        print('blended', sub, 'alpha', alpha, 'csls', csls_k)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'build':
        build(sys.argv[2], NEW_SPECS[:int(sys.argv[3])] if len(sys.argv) > 3 else NEW_SPECS)
    elif cmd == 'select':
        select_and_test()
    elif cmd == 'predict':
        predict(sys.argv[2], sys.argv[3], int(sys.argv[3][-1]))
