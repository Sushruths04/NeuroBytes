"""TEMPO-MIND test (17 Oct 2026): stream x latency-window scorers fused onto frozen J1.

Same outer/inner splits, seeds and candidate draws as scripts/run_nested_development.py, so `baseline`
reproduces that run and all variants are paired. One process per (subject, seed); then summarize:
    python scripts/run_tempo_mind.py --out ../outputs/mindread_tempo_<date> --subject 1 --seed 17
    python scripts/run_tempo_mind.py --out ../outputs/mindread_tempo_<date> --summarize
Training recordings of sub-01..04 only; no official test data, no labels, no external data.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'src'))
from mindread import features as fs  # noqa: E402
from mindread.data import FINAL_SUBJECTS, FirewallError, assert_development_subject, load_train  # noqa: E402
from mindread.eval import retrieval_draws, video_folds  # noqa: E402
from mindread.fusion import ResidualFusion, row_z  # noqa: E402
from mindread.methods import SeparateStreams, time_windows  # noqa: E402
from mindread.preprocess import target  # noqa: E402
from mindread.scoring import ranks_of_truth  # noqa: E402

_spec = importlib.util.spec_from_file_location('nested', HERE / 'scripts' / 'run_nested_development.py')
nested = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(nested)


def components(cfg):
    """{component name: (bands, target columns or None)} for every name used by a variant."""
    t0 = cfg['target']['t0_ms']; n_t = (cfg['target']['t1_ms'] - t0) // 10
    names = sorted({c for v in cfg['variants'].values() for c in v})
    out = {}
    for name in names:
        stream, win = name.split('@')
        cols = None
        if win != 'full':
            a, b = cfg['windows_ms'][win]
            cols = time_windows(17, n_t, [(a - t0) // 10, (b - t0) // 10])[0]
        out[name] = (cfg['streams'][stream], cols)
    return out


def make_fusion(cfg, variant):
    """Residual fusion for one variant; optional smoothness links adjacent windows of the same stream only."""
    names = cfg['variants'][variant]
    return ResidualFusion(names, l2=cfg['fusion_l2'], smooth=cfg.get('variant_smooth', {}).get(variant, 0.0),
                          groups=[n.split('@')[0] for n in names])


def run(cfg, out, subject, seed, final_test=False):
    # sub-05/06 TRAINING videos are the pre-registered final test (configs/frozen_T1_grid.json); never for tuning.
    if final_test:
        if subject not in FINAL_SUBJECTS:
            raise FirewallError('--final-test is only for sub-05/06')
    else:
        assert_development_subject(subject)
    root = HERE.parent / 'data/challenge_official/eeg-mind-reading-challenge'
    comps = components(cfg)
    all_bands = list(dict.fromkeys(cfg['baseline_bands'] + [b for bands, _ in comps.values() for b in bands]))
    tables = {b: {i: v for i, v in fs.load(b).items() if 1 <= i <= 1000} for b in all_bands}
    groups = {int(k): int(v) for k, v in json.loads((HERE / 'configs/dup_groups.json').read_text()).items() if int(k) <= 1000}
    make = lambda: SeparateStreams(cfg['baseline_bands'], {n: b for n, (b, _) in comps.items()},
                                   decoder_weight=cfg['decoder_weight'],
                                   stream_columns={n: c for n, (_, c) in comps.items() if c is not None},
                                   decoder_band_scale=cfg.get('decoder_band_scale', False))
    tr = load_train(str(root), subject)
    Y = target(tr.eeg, **cfg['target'])
    dw = cfg['decoder_weight']; start = time.monotonic()
    for fold, (a, h) in enumerate(video_folds(tr.ids, cfg['outer_folds'], seed, groups)):
        if (out / f'sub{subject:02d}_seed{seed}_fold{fold}.json').exists():
            continue                                   # resume: folds are independent and deterministic
        ia, ih = next(video_folds(tr.ids[a], cfg['inner_folds'], seed + 1009 + fold, groups))
        inner_train, calibration = a[ia], a[ih]
        if min(len(calibration), len(h)) < 102:
            raise ValueError('fold too small for 102-way retrieval')
        m = make().fit(tr.ids[inner_train], Y[inner_train], tables)
        inner = m.score_direction_components(Y[calibration], tables, tr.ids[calibration])
        draws = list(retrieval_draws(np.arange(len(calibration)), 102, cfg['draws'], seed + fold))
        subsets = [nested.subset_components(inner, c, dw) for c in draws]
        cal = {n: np.concatenate([d[n] for d in subsets]) for n in inner}
        truth = np.tile(np.arange(102), len(draws))
        fusions = {v: make_fusion(cfg, v).fit(cal, truth) for v in cfg['variants']}
        del m, inner, subsets, cal
        model = make().fit(tr.ids[a], Y[a], tables)
        raw = model.score_direction_components(Y[h], tables, tr.ids[h])
        evals = ['baseline', *cfg['variants']]
        ranks = {n: [] for n in evals}; queries = []
        for c in retrieval_draws(np.arange(len(h)), 102, cfg['draws'], seed + fold + 9001):
            comp = nested.subset_components(raw, c, dw)
            ranks['baseline'].append(ranks_of_truth(comp['baseline'], np.arange(102)))
            for v, f in fusions.items():
                ranks[v].append(ranks_of_truth(f.scores(comp), np.arange(102)))
            queries.append(tr.ids[h][c])
        qids = np.concatenate(queries); unique = np.unique(qids)
        qm = {}
        for n in evals:
            r = np.concatenate(ranks[n]); vals = np.stack([r == 1, r <= 5, r, 1 / r], axis=1)
            qm[n] = np.stack([vals[qids == q].mean(0) for q in unique]).tolist()
        rec = dict(subject=subject, seed=seed, fold=fold, calibration_ids=tr.ids[calibration].tolist(),
                   outer_held_ids=tr.ids[h].tolist(), query_ids=unique.tolist(), query_metrics=qm,
                   weights={v: dict(zip(cfg['variants'][v], f.weights.round(4).tolist())) for v, f in fusions.items()})
        name = f'sub{subject:02d}_seed{seed}_fold{fold}'
        (out / f'{name}.json').write_text(json.dumps(rec))
        print(name, {n: round(float(np.mean(qm[n], axis=0)[2]), 2) for n in evals},
              'elapsed_s', round(time.monotonic() - start), flush=True)
        del model, raw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=str(HERE / 'configs/development_tempo.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--subject', type=int); ap.add_argument('--seed', type=int)
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--final-test', action='store_true', help='run the frozen protocol once on sub-05/06 training videos')
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.final_test:
        cfg['subjects'] = list(FINAL_SUBJECTS)
    elif not cfg['subjects'] or not set(cfg['subjects']) <= {1, 2, 3, 4}:
        raise FirewallError('development subjects must be exclusively sub01..04')
    out = Path(args.out)
    if args.summarize:
        records = [json.loads(p.read_text()) for p in sorted(out.glob('sub*_fold*.json'))]
        n = len(cfg['subjects']) * len(cfg['seeds']) * cfg['outer_folds']
        if len(records) != n:
            raise SystemExit(f'{len(records)}/{n} folds present; refusing a partial summary')
        summary = nested.summarize(records, ['baseline', *cfg['variants']])
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: dict(mean=np.round(v['mean'], 4).tolist(), delta=np.round(v.get('paired_delta', []), 4).tolist())
                          for k, v in summary.items()}, indent=1))
        return
    if args.subject not in cfg['subjects'] or args.seed not in cfg['seeds']:
        raise SystemExit('need --subject and --seed from the config')
    out.mkdir(parents=True, exist_ok=True)
    mf = out / 'manifest.json'
    if not mf.exists():
        src = [Path(__file__), Path(args.config), HERE / 'scripts/run_nested_development.py',
               *sorted((HERE / 'src/mindread').glob('*.py'))]
        bands = sorted({b for s in cfg['streams'].values() for b in s} | set(cfg['baseline_bands']))
        mf.write_text(json.dumps(dict(config=cfg, source_hashes={str(p.resolve()): nested.sha(p) for p in src},
                                      features={b: nested.sha(Path(fs.CACHE) / f'{b}.npz') for b in bands}), indent=2))
    run(cfg, out, args.subject, args.seed, final_test=args.final_test)


if __name__ == '__main__':
    main()
