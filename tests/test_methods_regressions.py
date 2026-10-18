"""Synthetic regression checks; these are not dataset evaluations."""
import numpy as np
import pytest

from mindread.data import FirewallError
from mindread.methods import Decode, Encode, Fuse, SeparateStreams, TRF


def test_trf_pre_onset_lags_are_zero_and_do_not_wrap_to_clip_end():
    stimulus = np.arange(300, dtype=float)[None, :]
    model = TRF({1: stimulus}, t0_ms=500, t1_ms=700, fs=10,
                max_lag_ms=600, lag_step_ms=100)
    model.m = np.zeros((1, 1))
    model.s = np.ones((1, 1))
    design = model._design([1])
    np.testing.assert_array_equal(design[0], [50, 40, 30, 20, 10, 0, 0])
    np.testing.assert_array_equal(design[1], [60, 50, 40, 30, 20, 10, 0])
    stimulus[0, -10] = 1e9
    np.testing.assert_array_equal(model._design([1]), design)


def test_trf_padding_is_zero_after_standardization_at_both_boundaries():
    model = TRF({1: np.full((2, 300), 12.0)}, t0_ms=0, t1_ms=3100,
                fs=10, max_lag_ms=100, lag_step_ms=100)
    model.m = np.full((2, 1), 10.0)
    model.s = np.full((2, 1), 2.0)
    design = model._design([1])
    np.testing.assert_array_equal(design[0], [1, 0, 1, 0])
    np.testing.assert_array_equal(design[-1], [0, 1, 0, 1])


def _stream_fixture():
    rng = np.random.default_rng(72)
    ids = np.arange(1, 41)
    image = rng.normal(size=(45, 6))
    video = rng.normal(size=(45, 4))
    Y = image[:40] @ rng.normal(size=(6, 12)) + rng.normal(size=(40, 12)) * .1
    all_ids = np.r_[ids, np.arange(1001, 1006)]
    tables = {'image': dict(zip(all_ids, image)), 'video': dict(zip(all_ids, video))}
    return ids, Y, tables


def test_separate_streams_preserves_frozen_baseline_and_honors_target_columns():
    ids, Y, tables = _stream_fixture()
    columns = np.array([0, 2, 4, 6])
    model = SeparateStreams(['image'], {'early_video': ['video']},
                            stream_columns={'early_video': columns}).fit(ids, Y, tables)
    components = model.score_components(Y[:5], tables, ids[:5])
    baseline = Fuse(Encode(['image']), Decode(['image']), w=2.).fit(ids, Y, [tables['image']])
    np.testing.assert_array_equal(components['baseline'], baseline.scores(Y[:5], [tables['image']], ids[:5]))
    np.testing.assert_array_equal(model.scores(Y[:5], tables, ids[:5]), components['baseline'])
    expected_video = Fuse(Encode(['video']), Decode(['video']), w=2.).fit(ids, Y[:, columns], [tables['video']])
    np.testing.assert_array_equal(components['early_video'],
                                  expected_video.scores(Y[:5, columns], [tables['video']], ids[:5]))
    changed = Y[:5].copy()
    changed[:, [1, 3, 5, 7, 8, 9, 10, 11]] += 1e6
    np.testing.assert_array_equal(model.score_components(changed, tables, ids[:5])['early_video'],
                                  components['early_video'])


def test_separate_streams_fit_ignores_test_features_and_rejects_test_training_ids():
    ids, Y, tables = _stream_fixture()
    first = SeparateStreams(['image'], {'video': ['video']}).fit(ids, Y, tables)
    altered = {band: {i: f.copy() if i < 1001 else np.full_like(f, np.nan)
                      for i, f in table.items()} for band, table in tables.items()}
    second = SeparateStreams(['image'], {'video': ['video']}).fit(ids, Y, altered)
    for name in first.models:
        a, b = first.models[name], second.models[name]
        np.testing.assert_array_equal(a.enc.models[0][1].coef_, b.enc.models[0][1].coef_)
        np.testing.assert_array_equal(a.dec.r.coef_, b.dec.r.coef_)
        np.testing.assert_array_equal(a.enc.mu, b.enc.mu)
    with pytest.raises(FirewallError):
        SeparateStreams(['image'], {}).fit(np.arange(1001, 1041), Y, tables)


def test_separate_streams_validates_weights_and_column_indices():
    ids, Y, tables = _stream_fixture()
    with pytest.raises(ValueError, match='nonnegative'):
        SeparateStreams(['image'], {'video': ['video']}, weights={'video': -1})
    with pytest.raises(ValueError, match='invalid EEG target columns'):
        SeparateStreams(['image'], {'video': ['video']},
                        stream_columns={'video': [0, 12]}).fit(ids, Y, tables)


def test_raw_direction_cache_reconstructs_fusion_for_each_candidate_subset():
    ids, Y, tables = _stream_fixture()
    model = SeparateStreams(['image'], {'early_video': ['video']},
                            stream_columns={'early_video': [0, 2, 4, 6]}).fit(ids, Y, tables)
    raw = model.score_direction_components(Y[:10], tables, ids[:10])
    chosen = np.array([8, 2, 5, 0])
    direct = model.score_components(Y[chosen], tables, ids[chosen])
    z = lambda s: (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-12)
    for name, directions in raw.items():
        enc = directions['encoder'][np.ix_(chosen, chosen)]
        dec = directions['decoder'][np.ix_(chosen, chosen)]
        reconstructed = z(enc) + model.decoder_weight * z(dec)
        np.testing.assert_allclose(reconstructed, direct[name], rtol=1e-12, atol=1e-12)
