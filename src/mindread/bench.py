"""Run one configuration on both validation signals.

V1: 5-fold CV over training videos of subjects 1..4, 102-way retrieval on 20 random candidate sets per fold.
V2: fit on all training videos, score the official sub-01..04 test set (logged by eval.dev_validation).
"""
from __future__ import annotations

import numpy as np

from . import features as fs
from .data import TEST_IDS, VAL_SUBJECTS, _check_root, assert_development_subject, load_test, load_train
from .encoder import RidgeEncoder, corr_scores
from .eval import dev_validation, retrieval_draws, video_folds
from .preprocess import target
from .scoring import metrics, ranks_of_truth


def v1(root, subject, feat_table, make_model, tgt_kw, n_folds=5, n_draws=20, seed=0, groups=None):
    assert_development_subject(subject)
    tr = load_train(root, subject)
    Y = target(tr.eeg, **tgt_kw); F = fs.matrix(feat_table, tr.ids)
    ranks = []
    for k, (a, h) in enumerate(video_folds(tr.ids, n_folds, seed, groups)):
        m = make_model().fit(tr.ids[a], F[a], Y[a])
        P = m.predict(F[h])
        for cand in retrieval_draws(np.arange(len(h)), 102, n_draws, seed + k):
            S = corr_scores(Y[h][cand], P[cand])
            ranks.append(ranks_of_truth(S, np.arange(len(cand))))
    return metrics(np.concatenate(ranks))


def fit_full(root, subject, feat_table, make_model, tgt_kw):
    tr = load_train(root, subject)
    return make_model().fit(tr.ids, fs.matrix(feat_table, tr.ids), target(tr.eeg, **tgt_kw))


def test_scores(root, subject, model, feat_table, tgt_kw):
    """(102 recordings in original order, 102 candidates 1001..1102) score matrix."""
    Y = target(load_test(root, subject), **tgt_kw)
    return corr_scores(Y, model.predict(fs.matrix(feat_table, TEST_IDS)))


def v2(root, subject, model, feat_table, tgt_kw, config, exp_id):
    assert_development_subject(subject)
    return dev_validation(root, subject, test_scores(root, subject, model, feat_table, tgt_kw), config, exp_id)


def _others(root, subject, tgt_kw, allowed_ids):
    """Other subjects' TRAINING targets for the allowed (training-fold) video IDs: {vid: [targets]}."""
    assert_development_subject(subject)
    out = {}
    allowed = set(int(i) for i in allowed_ids)
    for k in VAL_SUBJECTS:
        if k == subject:
            continue
        tr = load_train(root, k); Y = target(tr.eeg, **tgt_kw)
        for i, y in zip(tr.ids, Y):
            if int(i) in allowed:
                out.setdefault(int(i), []).append(y)
    return out


def run_method(root, subject, make, tables, tgt_kw, post=None, n_folds=5, n_draws=20, seed=0, groups=None,
               use_others=False, config=None, exp_id=None, align=False):
    """V1 (always) and V2 (sub-01..04) for a methods.Encode-style object. `post` maps a score matrix -> score matrix."""
    assert_development_subject(subject)
    post = post or (lambda S: S)
    tr = load_train(root, subject); Y = target(tr.eeg, **tgt_kw)
    ranks = []
    for k, (a, h) in enumerate(video_folds(tr.ids, n_folds, seed, groups)):
        oth = (aligned_others(root, subject, tgt_kw, tr.ids[a]) if align else _others(root, subject, tgt_kw, tr.ids[a])) if use_others else None
        m = make().fit(tr.ids[a], Y[a], tables, oth)
        for cand in retrieval_draws(np.arange(len(h)), 102, n_draws, seed + k):
            hid = tr.ids[h][cand]
            S = post(m.scores(Y[h][cand], tables, hid))
            ranks.append(ranks_of_truth(S, np.arange(len(cand))))
    out = dict(v1=metrics(np.concatenate(ranks)))
    if subject <= 4:
        oth = (aligned_others(root, subject, tgt_kw, tr.ids) if align else _others(root, subject, tgt_kw, tr.ids)) if use_others else None
        m = make().fit(tr.ids, Y, tables, oth)
        S = post(m.scores(target(load_test(root, subject), **tgt_kw), tables, TEST_IDS))
        out['v2'] = dev_validation(root, subject, S, config or {}, exp_id or 'method')
    return out


_TRAIN_CACHE = {}


def _train_targets(root, k, tgt_kw):
    assert_development_subject(k)
    key = (_check_root(root), k, json_key(tgt_kw))
    if key not in _TRAIN_CACHE:
        tr = load_train(root, k); _TRAIN_CACHE[key] = (tr.ids, target(tr.eeg, **tgt_kw))
    return _TRAIN_CACHE[key]


def json_key(d):
    import json
    return json.dumps(d, sort_keys=True, default=str)


def aligned_others(root, subject, tgt_kw, fold_ids, n_pc=100):
    """N2: other subjects' TRAINING responses mapped into the target subject's space.

    For each other subject k: fit ridge Y_k -> Y_s on videos both saw within the target's training fold
    (PCA of Y_k to n_pc dims, fit on those videos), then map k's responses to all fold videos it saw.
    Returns {video_id: [mapped targets]} for fold videos only. Uses training recordings exclusively.
    """
    assert_development_subject(subject)
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    ids_s, Y_s = _train_targets(root, subject, tgt_kw)
    pos_s = {int(v): i for i, v in enumerate(ids_s)}
    fold = set(int(i) for i in fold_ids)
    out = {}
    for k in VAL_SUBJECTS:
        if k == subject:
            continue
        ids_k, Y_k = _train_targets(root, k, tgt_kw)
        pos_k = {int(v): i for i, v in enumerate(ids_k)}
        shared = [v for v in fold if v in pos_k and v in pos_s]
        if len(shared) < 200:
            continue
        A = Y_k[[pos_k[v] for v in shared]]; B = Y_s[[pos_s[v] for v in shared]]
        pca = PCA(n_components=min(n_pc, len(shared) - 1)).fit(A)
        r = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(pca.transform(A), B)
        vids = [v for v in fold if v in pos_k]
        M = r.predict(pca.transform(Y_k[[pos_k[v] for v in vids]]))
        for v, y in zip(vids, M):
            out.setdefault(v, []).append(y)
    return out
