import numpy as np
import pytest

from mindread.data import FirewallError, _check_root, assert_training_ids, load_test, load_train, load_val_labels
from mindread.eval import video_folds
from mindread.scoring import metrics, ranks_of_truth, to_ranked_ids, validate_submission


def test_external_paths_refused():
    for p in ['/x/data/external/derivatives/eeg', '/x/external', '/x/derivatives']:
        with pytest.raises(FirewallError):
            _check_root(p)


def test_no_labels_for_final_subjects(tmp_path):
    for s in (5, 6):
        with pytest.raises(FirewallError):
            load_val_labels(str(tmp_path), s)


def test_fit_guard_rejects_test_ids():
    assert_training_ids(np.arange(1, 1001))
    with pytest.raises(FirewallError):
        assert_training_ids([5, 1001])


@pytest.mark.parametrize('ids', [[], [np.nan], [np.inf], [1.5], [1.0], [True], ['1'], [[1]], [1, 1]])
def test_fit_guard_rejects_malformed_ids(ids):
    with pytest.raises(FirewallError):
        assert_training_ids(ids)


def test_firewall_resolves_root_and_file_symlinks_before_io(tmp_path, monkeypatch):
    hidden = tmp_path / 'external'
    hidden.mkdir()
    alias = tmp_path / 'allowed'
    alias.symlink_to(hidden, target_is_directory=True)
    with pytest.raises(FirewallError):
        _check_root(alias)
    official = tmp_path / 'official'
    (official / 'trn').mkdir(parents=True)
    (official / 'trn' / 'sub-01_trn.npz').symlink_to(hidden / 'synthetic.npz')
    monkeypatch.setattr(np, 'load', lambda *a, **kw: pytest.fail('forbidden file reached np.load'))
    with pytest.raises(FirewallError):
        load_train(official, 1)


@pytest.mark.parametrize('subject', [5, 6])
def test_sealed_subject_validation_rejected_before_io(subject, monkeypatch):
    from mindread import bench
    monkeypatch.setattr(bench, 'load_train', lambda *a, **kw: pytest.fail('sealed training data read'))
    monkeypatch.setattr(np, 'load', lambda *a, **kw: pytest.fail('sealed test data read'))
    calls = [
        lambda: bench.v1('/nonexistent', subject, {}, None, {}),
        lambda: bench.run_method('/nonexistent', subject, None, [], {}),
        lambda: bench.v2('/nonexistent', subject, None, {}, {}, {}, 'fixture'),
        lambda: bench._others('/nonexistent', subject, {}, [1]),
        lambda: bench.aligned_others('/nonexistent', subject, {}, [1]),
        lambda: load_test('/nonexistent', subject),
    ]
    for call in calls:
        with pytest.raises(FirewallError):
            call()


def test_final_inference_opt_in_and_refit_use_synthetic_files(tmp_path):
    (tmp_path / 'trn').mkdir()
    (tmp_path / 'tst').mkdir()
    np.savez(tmp_path / 'trn' / 'sub-05_trn.npz', **{'1': np.zeros((17, 10))})
    np.save(tmp_path / 'tst' / 'sub-05_tst.npy', np.zeros((2, 17, 10)))
    assert load_train(tmp_path, 5).subject == 5
    assert load_test(tmp_path, 5, final_inference=True).shape == (2, 17, 10)


def test_other_subject_helpers_do_not_load_final_subjects(monkeypatch):
    from mindread import bench
    from mindread.data import TrainSet
    seen = []
    def fake_load(root, subject):
        seen.append(subject)
        assert subject in (1, 2, 3, 4)
        return TrainSet(subject, np.array([1]), np.zeros((1, 2)))
    monkeypatch.setattr(bench, 'load_train', fake_load)
    monkeypatch.setattr(bench, 'target', lambda eeg, **kw: eeg)
    monkeypatch.setattr(bench, '_TRAIN_CACHE', {})
    bench._others('/synthetic', 1, {}, [1])
    assert seen == [2, 3, 4]
    seen.clear()
    bench.aligned_others('/synthetic', 1, {}, [1])
    assert seen == [1, 2, 3, 4]


def test_training_cache_is_scoped_to_root(monkeypatch, tmp_path):
    from mindread import bench
    from mindread.data import TrainSet
    seen = []
    def fake_load(root, subject):
        seen.append(str(root))
        return TrainSet(subject, np.array([1]), np.full((1, 2), len(seen)))
    monkeypatch.setattr(bench, 'load_train', fake_load)
    monkeypatch.setattr(bench, 'target', lambda eeg, **kw: eeg)
    monkeypatch.setattr(bench, '_TRAIN_CACHE', {})
    first = bench._train_targets(tmp_path / 'first', 1, {})[1]
    second = bench._train_targets(tmp_path / 'second', 1, {})[1]
    assert len(seen) == 2
    assert not np.array_equal(first, second)


def test_folds_disjoint_and_groups_kept_together():
    ids = np.arange(1, 101)
    groups = {10: 999, 20: 999}
    seen = []
    for tr, he in video_folds(ids, 5, 0, groups):
        assert not set(tr) & set(he)
        h = set(ids[he])
        assert (10 in h) == (20 in h)
        seen += list(he)
    assert sorted(seen) == list(range(100))


def test_ranking_and_format_fixture():  # synthetic fixture, not a result
    rng = np.random.default_rng(0)
    perm = rng.permutation(102)
    scores = rng.normal(size=(102, 102))
    scores[np.arange(102), perm] += 100      # perfect scorer
    r = ranks_of_truth(scores, perm)
    assert metrics(r)['top1'] == 1.0
    ranked = to_ranked_ids(scores)
    validate_submission(ranked)
    assert np.array_equal(ranked[:, 0], perm + 1001)   # row order preserved
    with pytest.raises(ValueError):
        validate_submission(ranked[:, :101])
