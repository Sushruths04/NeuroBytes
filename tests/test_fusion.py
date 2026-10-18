import numpy as np
from mindread.fusion import ResidualFusion, row_z


def test_calibration_prefers_informative_stream_and_rejects_adversary():
    rng = np.random.default_rng(41)
    n = 30
    truth = np.arange(n)
    good = np.eye(n) * 4 + rng.normal(0, .1, (n, n))
    c = dict(baseline=rng.normal(size=(n, n)), good=good, bad=-good)
    f = ResidualFusion(['good', 'bad']).fit(c, truth)
    assert f.weights[0] > 0.1
    assert f.weights[1] == 0
    weights = f.weights.copy()
    f.scores({k: v[::-1] for k, v in c.items()})
    np.testing.assert_array_equal(f.weights, weights)  # inference cannot retune
    assert np.mean(f.scores(c).argmax(1) == truth) > .9


def test_zero_weight_preserves_anchor_ranking_and_constant_stream_is_safe():
    rng = np.random.default_rng(7)
    anchor = rng.normal(size=(20, 20))
    c = dict(baseline=anchor, constant=np.ones_like(anchor))
    f = ResidualFusion(['constant']).fit(c, np.arange(20))
    assert f.weights[0] == 0
    np.testing.assert_array_equal(f.scores(c).argsort(1), anchor.argsort(1))
    assert np.isfinite(row_z(c['constant'])).all()
