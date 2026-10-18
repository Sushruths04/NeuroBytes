"""Validate a ranked candidate list; optionally score it on sub-01..04 (logged sanity check, not selection).

python scripts/check_submission.py submissions/C13
"""
import os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mindread.scoring import validate_submission  # noqa: E402

paths = []
for arg in sys.argv[1:]:                     # files, or folders containing the ranked candidate lists
    paths += sorted(os.path.join(arg, f) for f in os.listdir(arg) if f.endswith('.npy')) if os.path.isdir(arg) else [arg]
for p in paths:
    a = np.load(p)
    validate_submission(a)
    name = os.path.basename(p)
    if not (name.startswith(('sub-05_', 'sub-06_')) and name.endswith('_ranked_candidate_list.npy')):
        print(f'WARNING: {name} is not an official submission filename')
    print(f'OK {p}: {a.shape} {a.dtype}')
