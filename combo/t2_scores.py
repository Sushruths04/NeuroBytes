"""T2 (frozen fine7_swap procedure) as one scorer for COMBO, on winfuse's TRAIN/VAL split.

For development subjects writes a winfuse channel pickle 'T2:mindread:enc' with
  val  = T2 fitted on winfuse TRAIN only (fusion weights from 3 inner calibration splits inside TRAIN), VAL x VAL
  test = T2 fitted on all training recordings, dev test x 102 candidates (exactly the frozen procedure)
For sub-05/06 writes only the test matrix (final inference; never labels).
    python combo/t2_scores.py OUTDIR 1
"""
import importlib.util, json, os, pickle, sys

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

from mindread import features as fs  # noqa: E402
from mindread.data import TEST_IDS, load_test, load_train, load_val_labels  # noqa: E402
from mindread.eval import retrieval_draws, video_folds  # noqa: E402
from mindread.preprocess import target  # noqa: E402

M = MREPO
_s = importlib.util.spec_from_file_location('tempo', os.path.join(M, 'scripts', 'run_tempo_mind.py'))
tempo = importlib.util.module_from_spec(_s); _s.loader.exec_module(tempo)
from winfuse.compare_embed.pipeline import split_ids  # noqa: E402  (winfuse's fixed TRAIN/VAL split, seed 1234)

ROOT = DATA
cfg = json.load(open(os.path.join(M, 'configs', 'development_tempo_t2feat.json')))
VARIANT = 'fine7_swap'
comps = tempo.components(cfg)
groups = {int(k): int(v) for k, v in json.load(open(os.path.join(M, 'configs', 'dup_groups.json'))).items() if int(k) <= 1000}
make = lambda: tempo.SeparateStreams(cfg['baseline_bands'], {n: b for n, (b, _) in comps.items()},
                                     decoder_weight=cfg['decoder_weight'],
                                     stream_columns={n: c for n, (_, c) in comps.items() if c is not None})
dw = cfg['decoder_weight']


def t2(ids, Y, tables, Yq, cand_ids):
    """Frozen T2 procedure on training rows (ids, Y): 3-seed calibrated fusion weights, refit, score queries."""
    W = []
    for seed in cfg['seeds']:
        ia, ih = next(video_folds(ids, cfg['inner_folds'], seed + 1009, groups))
        m = make().fit(ids[ia], Y[ia], tables)
        inner = m.score_direction_components(Y[ih], tables, ids[ih])
        draws = list(retrieval_draws(np.arange(len(ih)), 102, cfg['draws'], seed))
        subsets = [tempo.nested.subset_components(inner, c, dw) for c in draws]
        cal = {n: np.concatenate([d[n] for d in subsets]) for n in inner}
        W.append(tempo.make_fusion(cfg, VARIANT).fit(cal, np.tile(np.arange(102), len(draws))).weights)
    fusion = tempo.make_fusion(cfg, VARIANT); fusion.weights = np.mean(W, 0)
    model = make().fit(ids, Y, tables)
    comp = tempo.nested.subset_components(model.score_direction_components(Yq, tables, cand_ids),
                                          np.arange(len(cand_ids)), dw)
    return fusion.scores(comp)


def main(out, s):
    bands = sorted({b for v in cfg['streams'].values() for b in v} | set(cfg['baseline_bands']))
    tables = {b: fs.load(b) for b in bands}
    tr = load_train(ROOT, s)
    order = np.argsort(tr.ids)                                    # winfuse orders trials by ascending video id
    ids, Y = tr.ids[order], target(tr.eeg, **cfg['target'])[order]
    Yt = target(load_test(ROOT, s, final_inference=s in (5, 6)), **cfg['target'])
    test = t2(ids, Y, tables, Yt, TEST_IDS).astype(np.float32)
    os.makedirs(out, exist_ok=True)
    if s in (5, 6):
        np.save(os.path.join(out, f'T2_test_sub-0{s}.npy'), test); print('test only', s, test.shape); return
    itr, iva = split_ids(list(ids))
    val = t2(ids[itr], Y[itr], tables, Y[iva], ids[iva]).astype(np.float32)
    truth = np.array([list(TEST_IDS).index(v) for v in load_val_labels(ROOT, s)])
    r = (np.argsort(-test, 1) == truth[:, None]).argmax(1) + 1
    rv = (np.argsort(-val, 1) == np.arange(len(val))[:, None]).argmax(1) + 1
    print(f'sub-0{s} T2 dev-test top1 {(r == 1).mean():.3f} top5 {(r <= 5).mean():.3f} | VAL ({len(val)}-way) top5 {(rv <= 5).mean():.3f}')
    pickle.dump({'val': val, 'test': test, 'truth': truth, 'spec': 'T2:mindread:enc', 'n_val': len(iva), 'n_train': len(itr)},
                open(os.path.join(out, f'sub-0{s}__T2_mindread_enc.pkl'), 'wb'))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
