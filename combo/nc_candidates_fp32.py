"""Rewrite <weights>/<tag>_candidates.pt with float32 candidate embeddings (exact reproduction of training-time scores).
    NC_TARGETS=... python combo/nc_candidates_fp32.py <weights_dir> <tag>"""
import os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import neural_clip as nc  # noqa: E402

d, tag = sys.argv[1], sys.argv[2]
C = torch.load(os.path.join(d, f'{tag}_candidates.pt'), weights_only=False)
tr = list(range(1, 1001)); te = list(nc.combo.P.TEST_IDS)
Ytr, Yte = nc.targets(tr), nc.targets(te)
C['cand'] = {t: torch.tensor(np.concatenate([Ytr[t], Yte[t]])) for t in C['targets']}
torch.save(C, os.path.join(d, f'{tag}_candidates.pt')); print('fp32 candidates', tag, {t: tuple(v.shape) for t, v in C['cand'].items()})
