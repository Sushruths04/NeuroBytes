"""Paired development ablation with inner calibration and sealed subjects 5/6.

python scripts/run_nested_development.py --out ../outputs/mindread_nested_<new>
No official test recordings or labels are accessed, including sub01..04.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'src'))
from mindread import features as fs
from mindread.data import load_train, FirewallError
from mindread.eval import video_folds, retrieval_draws
from mindread.fusion import ResidualFusion, row_z
from mindread.methods import SeparateStreams, time_windows
from mindread.preprocess import target
from mindread.scoring import ranks_of_truth


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def subset_components(raw, c, decoder_weight):
    """Normalize each direction within the actual 102-way candidate set."""
    return {name: row_z(parts['encoder'][np.ix_(c, c)])
                  + decoder_weight * row_z(parts['decoder'][np.ix_(c, c)])
            for name, parts in raw.items()}


def summarize(records, methods):
    out = {}
    for method in methods:
        by_subject = {}
        for s in sorted({r['subject'] for r in records}):
            seed_metrics = []
            for seed in sorted({r['seed'] for r in records}):
                rows = [r for r in records if r['subject'] == s and r['seed'] == seed]
                q = np.concatenate([np.asarray(r['query_metrics'][method]) for r in rows])
                seed_metrics.append(q.mean(0))
            by_subject[str(s)] = np.mean(seed_metrics, axis=0).tolist()
        out[method] = dict(metric_order=['top1', 'top5', 'mean_rank', 'mrr'],
                           by_subject=by_subject,
                           mean=np.mean(list(by_subject.values()), axis=0).tolist())
    for method in methods[1:]:
        delta = np.array([np.array(out[method]['by_subject'][s]) - out['baseline']['by_subject'][s]
                          for s in out[method]['by_subject']])
        mean = delta.mean(0)
        half = student_t.ppf(.975, len(delta)-1) * delta.std(0, ddof=1) / np.sqrt(len(delta))
        out[method]['paired_delta'] = mean.tolist()
        out[method]['paired_subject_t95'] = np.stack([mean-half, mean+half], axis=1).tolist()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=str(HERE / 'configs/development_v3.json'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if not cfg['subjects'] or not set(cfg['subjects']) <= {1, 2, 3, 4}:
        raise FirewallError('development subjects must be exclusively sub01..04')
    if cfg['candidate_count'] != 102:
        raise ValueError('the preregistered task is exactly 102-way retrieval')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    root = HERE.parent / 'data/challenge_official/eeg-mind-reading-challenge'
    bands = cfg['baseline_bands']
    all_bands = list(dict.fromkeys(bands + [cfg['video_band']]))
    # Feature-only caches may contain all candidates; never fit their test rows.
    tables = {b: {i: v for i, v in fs.load(b).items() if 1 <= i <= 1000} for b in all_bands}
    groups_path = HERE / 'configs/dup_groups.json'
    groups = {int(k): int(v) for k, v in json.loads(groups_path.read_text()).items() if int(k) <= 1000}
    source_paths = [Path(__file__), Path(args.config), groups_path,
                    *sorted((HERE / 'src/mindread').glob('*.py'))]
    manifest = dict(config=cfg, source_hashes={str(p.relative_to(HERE)): sha(p) for p in source_paths},
                    features={b: sha(Path(fs.CACHE) / f'{b}.npz') for b in all_bands},
                    input_hashes={}, protocol_note='new splits on previously used development data; not a fresh independent test')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    edges = [(x - cfg['target']['t0_ms']) // 10 for x in cfg['window_edges_ms']]
    columns = time_windows(17, edges[-1], edges)
    temporal = ['video_early', 'video_middle', 'video_late']
    streams = {n: [cfg['video_band']] for n in ['video_full', *temporal]}
    make = lambda: SeparateStreams(bands, streams, decoder_weight=cfg['decoder_weight'],
                                   stream_columns=dict(zip(temporal, columns)))
    records = []
    start = time.monotonic()
    for subject in cfg['subjects']:
        tr = load_train(str(root), subject)
        manifest['input_hashes'][f'sub-{subject:02d}_trn.npz'] = sha(root / 'trn' / f'sub-{subject:02d}_trn.npz')
        Y = target(tr.eeg, **cfg['target'])
        for seed in cfg['seeds']:
            for fold, (a, h) in enumerate(video_folds(tr.ids, cfg['outer_folds'], seed, groups)):
                # Calibration comes from the OUTER TRAINING partition only.
                ia, ih = next(video_folds(tr.ids[a], cfg['inner_folds'], seed + 1009 + fold, groups))
                inner_train, calibration = a[ia], a[ih]
                if min(len(calibration), len(h)) < 102:
                    raise ValueError('fold is too small for 102-way retrieval; refusing reduced-candidate scores')
                m = make().fit(tr.ids[inner_train], Y[inner_train], tables)
                inner_scores = m.score_direction_components(Y[calibration], tables, tr.ids[calibration])
                draws = list(retrieval_draws(np.arange(len(calibration)), 102, cfg['draws'], seed + fold))
                subsets = [subset_components(inner_scores, c, cfg['decoder_weight']) for c in draws]
                calibration_scores = {n: np.concatenate([d[n] for d in subsets]) for n in inner_scores}
                truth = np.tile(np.arange(102), len(draws))
                flat = ResidualFusion(['video_full'], l2=cfg['fusion_l2']).fit(calibration_scores, truth)
                gated = ResidualFusion(temporal, l2=cfg['fusion_l2'], smooth=cfg['fusion_smooth']).fit(calibration_scores, truth)
                del m, inner_scores, calibration_scores, subsets
                model = make().fit(tr.ids[a], Y[a], tables)
                raw = model.score_direction_components(Y[h], tables, tr.ids[h])
                ranks = {name: [] for name in cfg['evaluations']}
                queries = []
                for c in retrieval_draws(np.arange(len(h)), 102, cfg['draws'], seed + fold + 9001):
                    comp = subset_components(raw, c, cfg['decoder_weight'])
                    scores = dict(baseline=comp['baseline'],
                                  video_equal=row_z(comp['baseline']) + row_z(comp['video_full']),
                                  video_calibrated=flat.scores(comp), video_temporal=gated.scores(comp))
                    for name, S in scores.items():
                        ranks[name].append(ranks_of_truth(S, np.arange(102)))
                    queries.append(tr.ids[h][c])
                qids = np.concatenate(queries)
                unique = np.unique(qids)
                query_metrics = {}
                for name in ranks:
                    r = np.concatenate(ranks[name])
                    vals = np.stack([r == 1, r <= 5, r, 1/r], axis=1)
                    # Repeated candidate draws are Monte Carlo repeats, not independent observations.
                    query_metrics[name] = np.stack([vals[qids == q].mean(0) for q in unique]).tolist()
                record = dict(subject=subject, seed=seed, fold=fold,
                              inner_train_ids=tr.ids[inner_train].tolist(), calibration_ids=tr.ids[calibration].tolist(),
                              outer_train_ids=tr.ids[a].tolist(), outer_held_ids=tr.ids[h].tolist(),
                              flat_weights=flat.weights.tolist(), temporal_weights=gated.weights.tolist(),
                              query_ids=unique.tolist(), query_metrics=query_metrics)
                records.append(record)
                name = f'sub{subject:02d}_seed{seed}_fold{fold}'
                (out / f'{name}.json').write_text(json.dumps(record))
                np.savez_compressed(out / f'{name}_scores.npz', ids=tr.ids[h],
                                    **{f'{n}_{direction}': S for n, parts in raw.items() for direction, S in parts.items()})
                print(name, {n: round(float(np.mean(query_metrics[n], axis=0)[2]), 3) for n in ranks},
                      'elapsed_s', round(time.monotonic()-start), flush=True)
                del model, raw
    manifest['completed'] = True
    manifest['elapsed_seconds'] = time.monotonic() - start
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    summary = summarize(records, cfg['evaluations'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
