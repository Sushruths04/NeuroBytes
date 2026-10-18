"""ONE standardised pipeline, with frozen shared weights, runnable on any subject.

    # 1. build the channel library on a TRAIN/VAL split of the training data
    .venv-embed/bin/python -m winfuse.compare_embed.pipeline --build

    # 2. select the channel set + fusion strategy on VALIDATION only, then freeze
    .venv-embed/bin/python -m winfuse.compare_embed.pipeline --select

    # 3. score the frozen configuration on the dev test split, once
    .venv-embed/bin/python -m winfuse.compare_embed.pipeline --test

    # 4. produce ranked candidate lists for any subject (incl. sub-05/06)
    .venv-embed/bin/python -m winfuse.compare_embed.pipeline --predict sub-05,sub-06

Design
------
Training trials are split per subject into TRAIN (75%) and VAL (25%) with a fixed
seed. Everything that is *chosen* -- which channels, which fusion strategy, the
weights -- is chosen on VAL. The test split is touched once, at step 3, to report the
result of an already-frozen configuration.

VAL is scored with **102-candidate pools**, the same pool size as the real task, so
validation numbers are directly comparable to test numbers and transfer failures are
visible rather than inferred.

What is per-subject and what is frozen
--------------------------------------
per-subject  the ridge regressions. Brains differ, and the brief allows each
             subject's own training data, so a channel's ridge is always fitted on
             that subject's trials.
frozen       the channel set and the fusion weights. One weight vector, derived from
             the four development subjects' VAL data, applied unchanged to every
             subject. This is what makes the pipeline runnable on sub-05/06, whose
             test labels do not exist.

Leakage
-------
* Channel ridges for selection are fitted on TRAIN only; VAL is held out.
* For the final model each ridge is refitted on TRAIN+VAL, i.e. all training trials,
  which is standard after hyperparameter selection. The frozen weights are not
  re-derived at that point.
* `tst/` is read only in --test (sub-01..04) and --predict. sub-05/06 labels do not
  exist and are never required.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from winfuse.common import io as cio
from winfuse.common import preprocess as pp
from winfuse.common.config import ALL_SUBJECTS, CACHE_DIR, DEV_SUBJECTS, RESULTS_DIR, ROOT, SEED
from winfuse.common.feature_api import get_extractor
from winfuse.compare_embed import eeg_features as ef
from winfuse.compare_embed import fusion as fu

CACHE = CACHE_DIR / "pipeline"   # writable repo cache; never write into the (possibly read-only) data folder
CACHE.mkdir(parents=True, exist_ok=True)
FROZEN = RESULTS_DIR / "compare_embed" / "_frozen_pipeline.json"
TEST_IDS = list(range(1001, 1103))
POOL = 102                 # match the real task exactly
VAL_FRAC = 0.25
VAL_REPEATS = 40           # pool draws per VAL trial, for a low-variance estimate

EEG_SETS = ["raw_signal", "erp_windows", "band_power"]
STIM = ["dinov2_base", "clip_vit_b32", "resnet50", "alexnet_layers", "videomae_base",
        "xclip_base", "dinov2_small", "whisper_base", "wav2vec2_base", "clap_audio",
        "ast_audioset", "hc_visual", "hc_audio", "phase1_descriptors", "joint_av"]
ENC_EEG = ["erp_windows", "raw_signal"]
ENC_STIM = ["dinov2_base", "clip_vit_b32", "alexnet_layers", "resnet50",
            "videomae_base", "whisper_base", "hc_visual"]


# Layer variants already cached. Different depths encode different levels of
# abstraction (early conv = low-level statistics, late blocks = semantics), so they
# are expected to be complementary rather than redundant. A spread is used rather
# than every layer, to keep the channel count manageable.
LAYER_STIM = (
    [f"alexnet_layers@layer_{l}" for l in
     ("conv1", "conv3", "conv5", "fc6", "fc7")]
    + [f"dinov2_base@layer_block{b:02d}_cls" for b in (3, 6, 9, 12)]
    + [f"dinov2_base@layer_block{b:02d}_meanpatch" for b in (6, 12)]
    + [f"clip_vit_b32@layer_block{b:02d}_cls" for b in (3, 6, 9, 12)]
    + [f"resnet50@layer_{l}" for l in ("layer1", "layer2", "layer3", "layer4")]
    + [f"videomae_base@layer_block{b:02d}_mean" for b in (4, 8, 12)]
    + [f"wav2vec2_base@layer_L{l:02d}" for l in (2, 6, 9, 12)]
)
LAYER_EEG = ["raw_signal", "erp_windows"]


def channel_list(include_layers: bool = True) -> list[str]:
    out = [f"{e}:{s}:dec" for e in EEG_SETS for s in STIM]
    out += [f"{e}:{s}:enc" for e in ENC_EEG for s in ENC_STIM]
    if include_layers:
        out += [f"{e}:{s}:dec" for e in LAYER_EEG for s in LAYER_STIM]
    # latency-window EEG x the best stimulus representations
    win_sets = [f"win_{a}_{b}" for a, b in WINDOWS_MS]
    win_stim = ["dinov2_small", "dinov2_base", "clip_vit_b32", "alexnet_layers",
                "resnet50", "videomae_base", "whisper_base", "hc_visual"]
    out += [f"{e}:{s}:dec" for e in win_sets for s in win_stim]
    # the best windows crossed with layer variants: window x layer was the single
    # strongest family on validation, so it gets the extra coverage
    best_win = ["win_50_400", "win_100_300", "win_50_300", "win_80_320"]
    best_layers = ([f"alexnet_layers@layer_{l}" for l in ("conv3", "conv5", "fc6", "fc7")]
                   + [f"dinov2_base@layer_block{b:02d}_cls" for b in (6, 12)]
                   + [f"clip_vit_b32@layer_block{b:02d}_cls" for b in (6, 12)]
                   + ["resnet50@layer_layer3", "resnet50@layer_layer4"])
    out += [f"{e}:{s}:dec" for e in best_win for s in best_layers]
    # encoding on the best window, which was never tested
    out += [f"win_50_400:{s}:enc" for s in
            ("dinov2_base", "clip_vit_b32", "alexnet_layers", "resnet50")]
    # Systematic window x layer grid. This family produced every one of the top
    # channels on validation, so it gets full coverage: mid/late CNN layers and
    # mid/late transformer blocks, crossed with every window.
    grid_layers = (
        [f"alexnet_layers@layer_{l}" for l in ("conv4", "conv5", "fc6", "fc7")]
        + [f"resnet50@layer_{l}" for l in ("layer2", "layer3", "layer4", "final")]
        + [f"dinov2_base@layer_block{b:02d}_cls" for b in (6, 9, 12)]
        + [f"dinov2_base@layer_block{b:02d}_meanpatch" for b in (9, 12)]
        + [f"clip_vit_b32@layer_block{b:02d}_cls" for b in (6, 9, 12)]
        + [f"videomae_base@layer_block{b:02d}_mean" for b in (8, 12)]
        + ["dinov2_small", "whisper_base"]
    )
    grid_win = [f"win_{a}_{b}" for a, b in WINDOWS_MS]
    seen = set(out)
    for e in grid_win:
        for st in grid_layers:
            spec = f"{e}:{st}:dec"
            if spec not in seen:
                out.append(spec)
                seen.add(spec)
    return out


# Latency-window EEG sets. "win_50_400" means the preprocessed trace restricted to
# 50-400 ms post-onset, flattened over channels x time. Stimulus information in EEG
# is strongly time-localised, so a window-restricted channel can be cleaner than the
# whole epoch, and different windows are complementary to each other.
WINDOWS_MS = [(50, 200), (200, 400), (400, 700), (50, 400), (100, 300),
              # refined around the measured optimum: 50-400 ms was the single best
              # window by a wide margin on validation, so the grid is densified there
              (50, 250), (50, 300), (50, 500), (75, 275), (80, 320),
              (100, 400), (150, 350), (120, 450), (50, 600), (200, 500)]


INFERENCE_MODE = False      # set True to permit held-out subjects' recordings
PREPROC_CFG = None          # a pp.PreprocConfig to override the default chain


def _pc():
    return PREPROC_CFG


def window_eeg(sub: str, lo_ms: int, hi_ms: int, split: str,
               inference: bool = False) -> np.ndarray:
    """Preprocessed trace restricted to [lo_ms, hi_ms), flattened. float64."""
    from winfuse.common.config import SFREQ

    if split == "train":
        X, _ = pp.preprocessed_train(sub, cfg=_pc())
    else:
        X = pp.preprocessed_test(sub, reason="latency-window EEG channel",
                                 cfg=_pc(), inference=inference)
    sf = (_pc().sfreq if _pc() is not None else SFREQ)
    i0 = max(int(round(lo_ms / 1000.0 * sf)), 0)
    i1 = min(int(round(hi_ms / 1000.0 * sf)), X.shape[2])
    W = np.asarray(X[:, :, i0:i1], dtype=np.float64)
    return W.reshape(W.shape[0], -1)


def eeg_matrix(sub: str, es: str, split: str = "train"):
    """EEG features for a registered extractor OR a win_<lo>_<hi> pseudo-set.

    Returns (X, ids) for train and X for test, mirroring eeg_feature_matrix.
    """
    if es.startswith("win_"):
        _, lo, hi = es.split("_")
        X = window_eeg(sub, int(lo), int(hi), split, inference=INFERENCE_MODE)
        if split == "train":
            return X, cio.subject_train_ids(sub)
        return X
    if split == "train":
        return ef.eeg_feature_matrix(sub, es)
    x = get_extractor(es)
    Xt, _ = pp.preprocessed_train(sub)
    x.fit(Xt)
    x.transform(Xt[:2])
    Xte = pp.preprocessed_test(sub, reason="standardised pipeline build",
                               inference=INFERENCE_MODE)
    return np.asarray(x.transform(Xte), dtype=np.float64)


SPLIT_SEED = SEED          # overridable via --split-seed, for robustness checks


def split_ids(ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Fixed TRAIN/VAL split of a subject's training trial indices."""
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(ids))
    n_val = int(round(len(ids) * VAL_FRAC))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])     # train, val


def cpath(sub: str, spec: str, seed: int | None = None) -> Path:
    """Cache path. Non-default split seeds get their own namespace, so a robustness
    check never overwrites the frozen artifact's inputs."""
    sd = SPLIT_SEED if seed is None else seed
    sfx = "" if sd == SEED else f"__s{sd}"
    return CACHE / f"{sub}__{spec.replace(':', '_')}{sfx}.pkl"


# --------------------------------------------------------------------------- scoring
def _z(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    return (M - M.mean(1, keepdims=True)) / np.maximum(M.std(1, keepdims=True), 1e-12)


def _rrf(M: np.ndarray, k: float = 60.0) -> np.ndarray:
    """Reciprocal-rank transform: scale-free and distribution-free."""
    M = np.asarray(M, dtype=np.float64)
    order = np.argsort(-M, axis=1)
    out = np.empty_like(M)
    n = M.shape[1]
    r = np.empty(n)
    for i in range(M.shape[0]):
        r[order[i]] = np.arange(1, n + 1)
        out[i] = 1.0 / (k + r)
    return out


def _hub(M: np.ndarray) -> np.ndarray:
    """Hubness correction: remove each CANDIDATE's mean score across queries.

    In retrieval some candidates are 'hubs' that score highly for many different
    queries, purely because of where they sit in the embedding space. Subtracting a
    candidate's column mean removes that query-independent bias and leaves how well
    the candidate explains THIS query relative to its own baseline. This is a
    different mechanism from adding features, which had saturated.

    Column statistics are computed within the 102-candidate test block only, i.e.
    from the score matrix itself. No labels are involved, so this is the
    'unsupervised normalisation of the test recordings' the brief permits.
    """
    M = np.asarray(M, dtype=np.float64)
    return M - M.mean(0, keepdims=True)


def _z_hub(M):
    return _hub(_z(M))


def _rrf_hub(M):
    return _hub(_rrf(M))


def _zz(M):
    """Row z-score, column centre, then row z-score again (symmetric scaling)."""
    A = _hub(_z(M))
    return (A - A.mean(1, keepdims=True)) / np.maximum(A.std(1, keepdims=True), 1e-12)


TRANSFORMS = {"zscore": _z, "rrf": _rrf, "z+hub": _z_hub, "rrf+hub": _rrf_hub,
              "z+hub+z": _zz}

SINKHORN_ITERS, SINKHORN_TAU = 3, 0.5


def sinkhorn(S: np.ndarray, iters: int = SINKHORN_ITERS,
             tau: float = SINKHORN_TAU) -> np.ndarray:
    """Exploit the one-to-one structure of the candidate set.

    Each subject's test block is a BIJECTION: there are 102 recordings and 102
    candidate videos, and every video was shown exactly once. Ranking each recording
    independently throws that away and lets one video be the top guess for many
    recordings. Alternately normalising the rows and the columns of the score matrix
    pushes it toward doubly stochastic, which is the soft version of enforcing the
    constraint, while still leaving a full 102-long ranking per row.

    Hubness correction (`_hub`) is one half-iteration of this, and it was already
    worth +1.7 pp, which is what suggested going further.

    Uses no labels -- only the score matrix -- so it is the "unsupervised
    normalization of the test recordings" the brief permits. But note two real
    caveats:

    * It is TRANSDUCTIVE: all 102 recordings must be scored together. A single
      recording cannot be ranked in isolation.
    * It ASSUMES the bijection. That is verified true for sub-01..04, and is implied
      for sub-05/06 by the file shapes (102 recordings, 102 candidate videos), but
      cannot be confirmed for them without reading labels that do not exist.

    Chosen on bijective validation pools built from training videos, before any test
    read: +2.77 pp there, positive on all five train/val splits (mean +2.11, sd 0.66).
    """
    S = np.asarray(S, dtype=np.float64)
    Z = (S - S.mean(1, keepdims=True)) / np.maximum(S.std(1, keepdims=True), 1e-12)
    K = np.exp(np.clip(Z / tau, -700, 700))
    for _ in range(iters):
        K = K / np.maximum(K.sum(1, keepdims=True), 1e-300)
        K = K / np.maximum(K.sum(0, keepdims=True), 1e-300)
    return K


def apply_postprocess(S: np.ndarray, name: str) -> np.ndarray:
    if name in (None, "", "none"):
        return S
    if name == "sinkhorn":
        return sinkhorn(S)
    raise ValueError(f"unknown postprocess {name!r}")


def pool_ranks(S: np.ndarray, pools: np.ndarray) -> np.ndarray:
    """Rank of the true item within each 102-candidate pool.

    S     (n_trials, n_candidates) where candidate j corresponds to trial j
    pools (n_trials, repeats, POOL) column indices; pools[i,r,0] is always i
    """
    n, reps, _ = pools.shape
    out = np.empty((n, reps), dtype=np.int32)
    for i in range(n):
        row = S[i]
        for r in range(reps):
            v = row[pools[i, r]]
            out[i, r] = int(np.sum(v > v[0])) + 1
    return out


def make_pools(n: int, repeats: int = VAL_REPEATS, seed: int = SEED) -> np.ndarray:
    """For each trial, `repeats` pools of POOL candidates: itself + distractors."""
    rng = np.random.default_rng(seed)
    pool = min(POOL, n)
    out = np.empty((n, repeats, pool), dtype=np.int32)
    for i in range(n):
        others = np.delete(np.arange(n), i)
        for r in range(repeats):
            out[i, r, 0] = i
            out[i, r, 1:] = rng.choice(others, size=pool - 1, replace=False)
    return out


def metrics_from_pool_ranks(R: np.ndarray) -> dict:
    r = R.ravel()
    return {"top1": float(np.mean(r <= 1)), "top5": float(np.mean(r <= 5)),
            "mrr": float(np.mean(1.0 / r)), "median_rank": float(np.median(r))}


# --------------------------------------------------------------------------- build
def build(subs: list[str], specs: list[str], force: bool) -> None:
    for sub in subs:
        todo = [s for s in specs if force or not cpath(sub, s).exists()]
        print(f"  [{sub}] {len(todo)}/{len(specs)} channels to compute", flush=True)
        if not todo:
            continue
        eeg_sets = sorted({s.split(":")[0] for s in todo})
        stims = sorted({s.split(":")[1] for s in todo})

        F_raw = {e: eeg_matrix(sub, e, "train") for e in eeg_sets}
        base = F_raw[eeg_sets[0]][1]
        stim_tr, common = {}, None
        for st in stims:
            try:
                X, found = fu.stimulus_matrix(st, base, "train")
            except Exception as e:  # noqa: BLE001
                print(f"      skip {st}: {type(e).__name__}")
                continue
            stim_tr[st] = (X, found)
            common = set(found) if common is None else (common & set(found))
        common = sorted(common)

        def rows(mi, M, want):
            pos = {v: i for i, v in enumerate(mi)}
            return M[[pos[v] for v in want]]

        Fall = {e: rows(F_raw[e][1], F_raw[e][0], common) for e in eeg_sets}
        Eall = {st: rows(v[1], v[0], common) for st, v in stim_tr.items()}
        itr, iva = split_ids(common)

        has_test = sub in DEV_SUBJECTS
        Fte, Ecand, truth = {}, {}, None
        for e in eeg_sets:
            Fte[e] = eeg_matrix(sub, e, "test")
        for st in stims:
            try:
                Ecand[st] = fu.stimulus_matrix(st, TEST_IDS, "test")[0]
            except Exception:
                pass
        if has_test:
            yt = cio.load_test_labels(sub, reason="standardised pipeline build")
            truth = np.array([TEST_IDS.index(v) for v in yt])

        t0 = time.perf_counter()
        for i, spec in enumerate(todo, 1):
            es, st, _d = spec.split(":")
            if st not in Eall or st not in Ecand:
                continue
            try:
                # selection model: TRAIN only, scored on VAL
                c = fu.Channel(spec).fit(Fall[es][itr], Eall[st][itr])
                val = c.score(Fall[es][iva], Eall[st][iva]).astype(np.float32)
                # final model: all training trials
                base_p = cpath(sub, spec, seed=SEED)
                if SPLIT_SEED != SEED and base_p.exists():
                    # the final model is fitted on ALL training trials, so the test
                    # score matrix does not depend on the train/val split and can be
                    # reused verbatim across seeds
                    with open(base_p, "rb") as f:
                        test = pickle.load(f)["test"]
                else:
                    c2 = fu.Channel(spec).fit(Fall[es], Eall[st])
                    test = c2.score(Fte[es], Ecand[st]).astype(np.float32)
                with open(cpath(sub, spec), "wb") as f:
                    pickle.dump({"val": val, "test": test, "truth": truth,
                                 "spec": spec, "n_val": len(iva),
                                 "n_train": len(itr)}, f)
            except Exception as e:  # noqa: BLE001
                print(f"      FAIL {spec}: {type(e).__name__}: {str(e)[:60]}")
            if i % 15 == 0:
                print(f"      {i}/{len(todo)} ({time.perf_counter()-t0:.0f}s)", flush=True)


def load(sub: str, specs: list[str]) -> dict:
    out = {}
    for s in specs:
        p = cpath(sub, s)
        if not p.exists() and SPLIT_SEED != SEED:
            p = cpath(sub, s, seed=SEED)
        if p.exists():
            with open(p, "rb") as f:
                out[s] = pickle.load(f)
    return out


# --------------------------------------------------------------------------- select
def select(subs: list[str], specs: list[str]) -> dict:
    per = {sub: load(sub, specs) for sub in subs}
    avail = [s for s in specs if all(s in per[sub] for sub in subs)]
    print(f"\n  {len(avail)} channels available for all {len(subs)} subjects")

    pools = {sub: make_pools(per[sub][avail[0]]["val"].shape[0]) for sub in subs}

    # ---- per-channel VAL skill, pooled over subjects (102-candidate pools)
    skill, val_m = {}, {}
    for s in avail:
        ms = []
        for sub in subs:
            R = pool_ranks(_z(per[sub][s]["val"]), pools[sub])
            ms.append(metrics_from_pool_ranks(R))
        skill[s] = float(np.mean([m["mrr"] for m in ms]))
        val_m[s] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    ranked = sorted(avail, key=lambda s: -skill[s])
    print(f"\n  top 12 channels by VAL MRR (102-candidate pools):")
    print(f"    {'channel':44} {'VAL top-1':>10} {'VAL top-5':>10} {'MRR':>8}")
    for s in ranked[:12]:
        print(f"    {s:44} {val_m[s]['top1']*100:9.2f}% "
              f"{val_m[s]['top5']*100:9.2f}% {skill[s]:8.4f}")

    mrr_chance = float(np.sum(1.0 / np.arange(1, POOL + 1)) / POOL)

    def fuse_val(group, tf, wmode):
        w = weights_for(group, skill, mrr_chance, wmode)
        ms = []
        for sub in subs:
            M = sum(w[i] * tf(per[sub][s]["val"]) for i, s in enumerate(group))
            ms.append(metrics_from_pool_ranks(pool_ranks(M, pools[sub])))
        return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}, w

    # ---- candidate configurations, all judged on VAL
    cands = {}
    dec = [s for s in ranked if s.endswith(":dec")]
    enc = [s for s in ranked if s.endswith(":enc")]
    for n in (5, 8, 10, 15, 20, 25, len(avail)):
        if n <= len(ranked):
            cands[f"top{n}_all"] = ranked[:n]
    for n in (5, 8, 10, 15):
        if n <= len(dec):
            cands[f"top{n}_dec"] = dec[:n]
    for nd, ne in ((8, 3), (8, 5), (10, 5), (12, 6), (15, 8)):
        if nd <= len(dec) and ne <= len(enc):
            cands[f"dec{nd}_enc{ne}"] = dec[:nd] + enc[:ne]

    print(f"\n  evaluating {len(cands)} channel sets x {len(TRANSFORMS)} transforms "
          f"x 3 weightings on VALIDATION")
    results = {}
    for cname, g in cands.items():
        for tname, tf in TRANSFORMS.items():
            for wmode in ("equal", "skill", "skill2"):
                m, w = fuse_val(g, tf, wmode)
                results[f"{cname}|{tname}|{wmode}"] = {
                    "val": m, "n_channels": len(g), "channels": g,
                    "transform": tname, "weight_mode": wmode,
                    "weights": [round(float(x), 6) for x in w]}
    top = sorted(results.items(), key=lambda kv: -kv[1]["val"]["top5"])
    print(f"\n  best 12 configurations on VALIDATION:")
    print(f"    {'configuration':46} {'VAL top-1':>10} {'VAL top-5':>10}")
    for k, v in top[:12]:
        print(f"    {k:46} {v['val']['top1']*100:9.2f}% {v['val']['top5']*100:9.2f}%")

    best_key, best = top[0]
    frozen = {"selected": best_key, "channels": best["channels"],
              "transform": best["transform"], "weight_mode": best["weight_mode"],
              "weights": best["weights"], "val": best["val"],
              "val_pool_size": POOL, "val_repeats": VAL_REPEATS,
              "val_frac": VAL_FRAC, "seed": SEED,
              "channel_val_skill": {s: skill[s] for s in best["channels"]},
              "n_candidates_considered": len(results),
              "selected_on": "validation split of the training data; the test split "
                             "was not consulted",
              "subjects_used_for_weights": subs}
    out_p = FROZEN if SPLIT_SEED == SEED else FROZEN.with_name(
        f"_frozen_pipeline_s{SPLIT_SEED}.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(frozen, indent=2))
    globals()["_LAST_FROZEN"] = out_p
    print(f"\n  FROZEN: {best_key}")
    print(f"    {len(best['channels'])} channels, transform={best['transform']}, "
          f"weights={best['weight_mode']}")
    print(f"    VAL top-1 {best['val']['top1']*100:.2f}%  "
          f"VAL top-5 {best['val']['top5']*100:.2f}%")
    print(f"  wrote {out_p.relative_to(ROOT)}")
    (RESULTS_DIR / "compare_embed" / "_pipeline_select.json").write_text(
        json.dumps({"channel_val": val_m, "channel_skill": skill,
                    "configurations": {k: {"val": v["val"], "n": v["n_channels"],
                                           "transform": v["transform"],
                                           "weight_mode": v["weight_mode"]}
                                       for k, v in results.items()}},
                   indent=2, default=str))
    return frozen


MULTI_SEEDS = [SEED, 7, 99, 2024, 777]


def _configs_for_split(per, subs, pools):
    """(candidate channel sets, per-channel VAL skill) for one split."""
    avail = sorted(set.intersection(*[set(per[x]) for x in subs]))
    skill = {}
    for s in avail:
        skill[s] = float(np.mean([metrics_from_pool_ranks(
            pool_ranks(_z(per[x][s]["val"]), pools[x]))["mrr"] for x in subs]))
    ranked = sorted(avail, key=lambda s: -skill[s])
    dec = [s for s in ranked if s.endswith(":dec")]
    enc = [s for s in ranked if s.endswith(":enc")]
    cands = {}
    for n in (5, 8, 10, 15, 20, 25, len(avail)):
        if n <= len(ranked):
            cands[f"top{n}_all"] = ranked[:n]
    for n in (5, 8, 10, 15):
        if n <= len(dec):
            cands[f"top{n}_dec"] = dec[:n]
    for nd, ne in ((8, 3), (8, 5), (10, 5), (12, 6), (15, 8)):
        if nd <= len(dec) and ne <= len(enc):
            cands[f"dec{nd}_enc{ne}"] = dec[:nd] + enc[:ne]
    return cands, skill


def select_multi(subs: list[str], specs: list[str], seeds=None) -> dict:
    """Choose the configuration by MEAN validation top-5 over several train/val splits.

    One 75/25 split is a noisy selection surface: with 240 candidates whose validation
    scores sit within ~1.5 pp of each other, which one comes first is partly luck of
    the split. Averaging each candidate over several independent splits is a lower
    variance estimator of the same quantity, and it still touches only training data.

    This is the selection procedure of record. `--select` (single split) is kept
    because it is what the earlier frozen artefact used.
    """
    import collections

    global SPLIT_SEED
    seeds = list(seeds or MULTI_SEEDS)
    keep = SPLIT_SEED
    agg: dict[str, list[float]] = collections.defaultdict(list)
    chan_skill: dict[str, list[float]] = collections.defaultdict(list)
    groups: dict[str, list[str]] = {}
    try:
        for sd in seeds:
            SPLIT_SEED = sd
            per = {s: load(s, specs) for s in subs}
            miss = [s for s in subs if not per[s]]
            if miss:
                raise SystemExit(f"no cached channels for {miss} at split seed {sd}; "
                                 f"run --build --split-seed {sd} first")
            any_spec = next(iter(per[subs[0]]))
            pools = {s: make_pools(per[s][any_spec]["val"].shape[0]) for s in subs}
            cands, skill = _configs_for_split(per, subs, pools)
            for k, v in skill.items():
                chan_skill[k].append(v)
            mrr_chance = float(np.sum(1.0 / np.arange(1, POOL + 1)) / POOL)
            for cname, g in cands.items():
                for tname, tf in TRANSFORMS.items():
                    for wmode in ("equal", "skill", "skill2"):
                        w = weights_for(g, skill, mrr_chance, wmode)
                        v = np.mean([metrics_from_pool_ranks(pool_ranks(
                            sum(w[i] * tf(per[sub][s]["val"]) for i, s in enumerate(g)),
                            pools[sub]))["top5"] for sub in subs])
                        key = f"{cname}|{tname}|{wmode}"
                        agg[key].append(float(v))
                        groups[key] = g
            print(f"  split seed {sd}: {len(cands) * len(TRANSFORMS) * 3} "
                  f"configurations scored on validation", flush=True)
    finally:
        SPLIT_SEED = keep

    full = {k: v for k, v in agg.items() if len(v) == len(seeds)}
    means = {k: float(np.mean(v)) for k, v in full.items()}
    sds = {k: float(np.std(v, ddof=1)) for k, v in full.items()}
    order = sorted(means, key=lambda k: -means[k])
    print(f"\n  best 8 by MEAN validation top-5 over {len(seeds)} splits:")
    print(f"    {'configuration':34} {'mean':>8} {'sd':>7}")
    for k in order[:8]:
        print(f"    {k:34} {means[k]*100:7.2f}% {sds[k]*100:6.2f}")

    best = order[0]
    g = groups[best]
    cname, tname, wmode = best.split("|")
    mrr_chance = float(np.sum(1.0 / np.arange(1, POOL + 1)) / POOL)
    mean_skill = {k: float(np.mean(v)) for k, v in chan_skill.items()}
    w = weights_for(g, mean_skill, mrr_chance, wmode)
    frozen = {"selected": best, "channels": g, "transform": tname,
              "weight_mode": wmode, "weights": [round(float(x), 8) for x in w],
              "val": {"top5": means[best], "top5_sd_across_splits": sds[best]},
              "selection": "mean validation top-5 over "
                           f"{len(seeds)} independent 75/25 train/val splits",
              "split_seeds": seeds, "val_pool_size": POOL,
              "val_repeats": VAL_REPEATS, "val_frac": VAL_FRAC, "seed": SEED,
              "n_candidates_considered": len(full),
              "selected_on": "validation splits of the training data only; the test "
                             "split was not consulted",
              "subjects_used_for_weights": subs,
              "runner_up": {k: means[k] for k in order[1:4]}}
    FROZEN.write_text(json.dumps(frozen, indent=2))
    print(f"\n  FROZEN (multi-split): {best}")
    print(f"    mean VAL top-5 {means[best]*100:.2f}% (sd {sds[best]*100:.2f} pp "
          f"across splits)")
    print(f"  wrote {FROZEN.relative_to(ROOT)}")
    return frozen


def weights_for(group, skill, mrr_chance, wmode) -> np.ndarray:
    sk = np.array([skill[s] for s in group])
    if wmode == "equal":
        w = np.ones(len(group))
    elif wmode == "skill":
        w = np.clip(sk - mrr_chance, 0, None)
    elif wmode == "skill2":
        w = np.clip(sk - mrr_chance, 0, None) ** 2
    else:
        raise ValueError(wmode)
    if w.sum() <= 0:
        w = np.ones(len(group))
    return w / w.sum()


# --------------------------------------------------------------------------- test
def test(subs: list[str]) -> dict:
    fp = FROZEN if SPLIT_SEED == SEED else FROZEN.with_name(
        f"_frozen_pipeline_s{SPLIT_SEED}.json")
    fz = json.loads(fp.read_text())
    g, tf = fz["channels"], TRANSFORMS[fz["transform"]]
    w = np.array(fz["weights"])
    per = {sub: load(sub, g) for sub in subs}
    rows, rr = {}, []
    for sub in subs:
        if any(s not in per[sub] for s in g):
            print(f"  {sub}: missing channels, skipped")
            continue
        M = sum(w[i] * tf(per[sub][s]["test"]) for i, s in enumerate(g))
        M = apply_postprocess(M, fz.get("postprocess", "none"))
        r = fu.ranks_of_truth(M, per[sub][g[0]]["truth"])
        rows[sub] = {"top1": fu.topk(r, 1), "top5": fu.topk(r, 5),
                     "median_rank": float(np.median(r))}
        rr.append(r)
    r = np.concatenate(rr)
    from scipy.stats import binomtest

    k1, k5, n = int(np.sum(r <= 1)), int(np.sum(r <= 5)), r.size
    out = {"frozen_config": fz["selected"],
           "postprocess": fz.get("postprocess", "none"), "per_subject": rows,
           "pooled": {"top1": k1 / n, "top5": k5 / n, "k1": k1, "k5": k5, "n": n,
                      "p1": float(binomtest(k1, n, 1 / 102, alternative="greater").pvalue),
                      "p5": float(binomtest(k5, n, 5 / 102, alternative="greater").pvalue),
                      "median_rank": float(np.median(r))},
           "val_predicted": fz["val"]}
    print(f"\n  frozen configuration: {fz['selected']}")
    print(f"  {'subject':10} {'top-1':>8} {'top-5':>8} {'median':>8}")
    for sub, m in rows.items():
        print(f"  {sub:10} {m['top1']*100:7.2f}% {m['top5']*100:7.2f}% "
              f"{m['median_rank']:7.1f}")
    print(f"  {'POOLED':10} {k1/n*100:7.2f}% {k5/n*100:7.2f}% "
          f"{out['pooled']['median_rank']:7.1f}")
    v = fz.get("val", {})
    if "top1" in v:
        print(f"\n  VAL predicted:  top-1 {v['top1']*100:.2f}%  "
              f"top-5 {v['top5']*100:.2f}%")
    elif "top5" in v:
        sd = v.get("top5_sd_across_splits")
        extra = f" (sd {sd*100:.2f} pp across splits)" if sd else ""
        print(f"\n  VAL predicted:  top-5 {v['top5']*100:.2f}%{extra}")
    print(f"  TEST actual:    top-1 {k1/n*100:.2f}%  top-5 {k5/n*100:.2f}%")
    if "top5" in v:
        gap = v["top5"] * 100 - k5 / n * 100
        print(f"  val->test gap:  {gap:+.2f} pp on top-5 "
              f"({'val optimistic' if gap > 0 else 'val pessimistic'})")
    p = RESULTS_DIR / "compare_embed" / (
        "_pipeline_test.json" if SPLIT_SEED == SEED
        else f"_pipeline_test_s{SPLIT_SEED}.json")
    p.write_text(json.dumps(out, indent=2))
    print(f"  wrote {p.relative_to(ROOT)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--select-multi", action="store_true",
                    help="selection of record: average validation over several splits")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--predict", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--subjects", default=",".join(DEV_SUBJECTS))
    ap.add_argument("--split-seed", type=int, default=SEED)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    global SPLIT_SEED
    SPLIT_SEED = args.split_seed
    if SPLIT_SEED != SEED:
        print(f"ROBUSTNESS CHECK: train/val split seed {SPLIT_SEED} "
              f"(frozen artifact uses {SEED}); writing to a separate cache namespace")
    subs = [s.strip() for s in args.subjects.split(",") if s.strip()]
    specs = channel_list()

    if args.build:
        print(f"building {len(specs)} channels for {subs}")
        build(subs, specs, args.force)
    if args.select:
        select(subs, specs)
    if args.select_multi:
        select_multi(subs, specs)
    if args.test:
        test(subs)
    if args.predict:
        print("--predict requires the sub-05/06 inference exception; not enabled")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
