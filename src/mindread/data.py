"""Data access with a hard firewall between training data, validation labels and final test data.

Rules enforced here (challenge PDF + organizer intro video):
- Development uses subjects 1..4; training recordings of all 6 may be fitted on after freezing for final inference.
- sub-01..04 test labels: validation and model selection (PDF), via the logged `eval.dev_validation`.
- sub-05/06 test recordings require explicit final inference; no labels exist for them in this code base.
- Folders of external datasets (`external/`, BIDS `derivatives/`) are refused; this is a path heuristic and a guard, not an OS sandbox.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

CHANNELS = ('Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'P1', 'P5',
            'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2')  # channel order as provided by the organizers
SFREQ = 500
TEST_IDS = np.arange(1001, 1103)
VAL_SUBJECTS = (1, 2, 3, 4)
FINAL_SUBJECTS = (5, 6)


class FirewallError(RuntimeError):
    pass


EXTERNAL_DIRS = ('external', 'derivatives')


def _check_root(root: str) -> str:
    # Check both the supplied path and its resolved target, including file symlinks.
    for path in (os.path.abspath(root), os.path.realpath(root)):
        parts = [p.lower() for p in os.path.normpath(path).split(os.sep)]
        if any(p in EXTERNAL_DIRS or p.startswith('external_') for p in parts):
            raise FirewallError(f'refusing to read from {root}: only the official challenge data may be used')
    return os.path.realpath(root)


def assert_development_subject(subject: int) -> None:
    """Final subjects are sealed against validation and model selection."""
    if isinstance(subject, (bool, np.bool_)) or not isinstance(subject, (int, np.integer)) or subject not in VAL_SUBJECTS:
        raise FirewallError(f'subject {subject!r} is not a development subject (1..4)')


def _subject_file(root, folder, subject, suffix):
    if isinstance(subject, (bool, np.bool_)) or not isinstance(subject, (int, np.integer)) or subject not in (*VAL_SUBJECTS, *FINAL_SUBJECTS):
        raise FirewallError(f'invalid subject {subject!r}')
    return _check_root(os.path.join(_check_root(root), folder, f'sub-{subject:02d}_{suffix}'))


@dataclass(frozen=True)
class TrainSet:
    subject: int
    ids: np.ndarray      # (n,) training video IDs, 1..1000
    eeg: np.ndarray      # (n, 17, 1500), read-only


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.setflags(write=False)
    return a


def load_train(root: str, subject: int) -> TrainSet:
    z = np.load(_subject_file(root, 'trn', subject, 'trn.npz'))
    ids = np.array(sorted(int(k) for k in z.files))
    assert_training_ids(ids)
    eeg = np.stack([z[str(i)] for i in ids]).astype(np.float32)
    return TrainSet(subject, _readonly(ids), _readonly(eeg))


def load_test(root: str, subject: int, *, final_inference: bool = False) -> np.ndarray:
    """Test recordings in original order; subjects 5/6 require an explicit final inference call."""
    if subject in FINAL_SUBJECTS and final_inference is not True:
        raise FirewallError('sub-05/06 test recordings are sealed; freeze the model before final inference')
    return _readonly(np.load(_subject_file(root, 'tst', subject, 'tst.npy')).astype(np.float32))


def load_val_labels(root: str, subject: int) -> np.ndarray:
    """sub-01..04 only. Use through mindread.eval.dev_validation (logged). Never as training targets or templates."""
    assert_development_subject(subject)
    return _readonly(np.load(_subject_file(root, 'tst_stim', subject, 'tst_stim.npy')))


def assert_training_ids(ids) -> None:
    """Call at the top of every fit(): fitting on any test video ID is a bug."""
    ids = np.asarray(ids)
    if ids.ndim != 1 or not ids.size or not np.issubdtype(ids.dtype, np.integer):
        raise FirewallError('fit() requires a nonempty vector of integer training video IDs')
    if np.any(ids > 1000) or np.any(ids < 1) or len(np.unique(ids)) != len(ids):
        raise FirewallError('fit() received invalid, duplicate or test video IDs')
