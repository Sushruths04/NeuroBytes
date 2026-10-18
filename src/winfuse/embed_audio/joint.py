"""Joint audio-visual embedding, built from cached pooled embeddings.

`joint_av` is the concatenation of the best available VISUAL pooled embedding and the
best available AUDIO pooled embedding, each z-scored with mean/std fitted on the
TRAINING videos (IDs <= 1000) only. This is a cache-derived product, not a per-video
network, so it is NOT registered as a BaseEmbedder — V-C reads it straight from
`cache/embeddings/joint_av/` like any other consolidated model.

Visual source selection (in priority order), to cooperate with Agent V-A running in
parallel without reimplementing its models:
    1. a V-A visual model if its pooled cache exists (prefer dinov2_base > clip_vit_b32
       > videomae_base > xclip_base > any available visual model)
    2. otherwise our own `hc_visual`
Audio source: `wav2vec2_base` pooled if available, else `hc_audio`.

The exact sources chosen, the concatenation order (visual first, then audio) and the
z-scoring are logged with log_decision and written into the manifest.
"""
from __future__ import annotations

import time

import numpy as np

from winfuse.common import embed_cache as ec
from winfuse.common.config import SEED
from winfuse.common.decisions import log_decision

from . import constants as K

np.random.seed(SEED)

VISUAL_PRIORITY = ["dinov2_base", "clip_vit_b32", "videomae_base", "xclip_base_patch32",
                   "dinov2_small", "resnet50", "hc_visual"]
AUDIO_PRIORITY = ["wav2vec2_base", "clap_audio", "ast_audioset", "whisper_base",
                  "whisper_tiny", "hc_audio"]

JOINT_NAME = "joint_av"

# A source must cover at least this many videos to be chosen; otherwise an in-progress
# parallel agent's partial cache (e.g. 20 videos) would cap the joint. 1102 = all.
MIN_COVERAGE = 1102


def _first_available(priority: list[str], min_coverage: int = MIN_COVERAGE) -> str | None:
    """Highest-priority model whose pooled cache covers >= min_coverage videos."""
    models = set(ec.list_models())
    best_partial = None
    for m in priority:
        if m in models and "pooled" in ec.available(m):
            try:
                X, ids = ec.load(m, "pooled")
            except Exception:
                continue
            if X.ndim == 2 and X.shape[0] >= min_coverage:
                return m
            if best_partial is None and X.ndim == 2 and X.shape[0] > 0:
                best_partial = m
    # nothing fully covers; fall back to the best partial so we still build something
    return best_partial


def _zscore_train(X: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, dict]:
    """Z-score columns using statistics fitted on training videos only (ids <= 1000)."""
    mask = ids <= 1000
    mu = X[mask].mean(0, keepdims=True)
    sd = X[mask].std(0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - mu) / sd
    return Z.astype(np.float32), {"n_train_fit": int(mask.sum())}


def build_joint(verbose: bool = True) -> dict:
    """Consolidate joint_av from the best available visual + audio pooled caches.

    Returns a status dict. Never raises if a source is missing — it logs a decision and
    reports what it did instead.
    """
    t0 = time.perf_counter()
    vis = _first_available(VISUAL_PRIORITY)
    aud = _first_available(AUDIO_PRIORITY)

    if aud is None:
        log_decision(
            agent="V-B", tags=["phase2", "joint", "blocked"],
            title="Joint embedding skipped — no audio pooled cache",
            choice="joint_av not built; no audio model has a consolidated pooled cache yet.",
            alternatives=["wait for an audio model to finish (resume-safe, re-run later)"],
            rationale="The joint model concatenates audio and visual; without an audio "
                      "cache there is nothing audio-side to join.",
            consequences="joint_av absent until at least one audio model is consolidated.",
        )
        return {"built": False, "reason": "no audio cache", "visual": vis, "audio": None}
    if vis is None:
        log_decision(
            agent="V-B", tags=["phase2", "joint", "blocked"],
            title="Joint embedding skipped — no visual pooled cache",
            choice="joint_av not built; neither a V-A visual cache nor hc_visual is present.",
            alternatives=["run hc_visual first (owned by V-B), then rebuild joint"],
            rationale="Visual side missing; V-A may still be running in parallel.",
            consequences="joint_av absent until a visual cache (ideally V-A's, else hc_visual) exists.",
        )
        return {"built": False, "reason": "no visual cache", "visual": None, "audio": aud}

    Xv, idv = ec.load(vis, "pooled")
    Xa, ida = ec.load(aud, "pooled")

    # align on the intersection of covered IDs, ascending
    common = np.array(sorted(set(idv.tolist()) & set(ida.tolist())), dtype=np.int32)
    pv = {int(v): i for i, v in enumerate(idv)}
    pa = {int(v): i for i, v in enumerate(ida)}
    Xv = Xv[[pv[int(v)] for v in common]]
    Xa = Xa[[pa[int(v)] for v in common]]

    Zv, sv = _zscore_train(Xv, common)
    Za, sa = _zscore_train(Xa, common)
    Z = np.concatenate([Zv, Za], axis=1).astype(np.float32)  # visual first, then audio

    # write directly as a consolidated model via per-video parts, then consolidate
    for i, vid in enumerate(common):
        ec.save_part(JOINT_NAME, "pooled", int(vid), Z[i])
    cons = ec.consolidate(JOINT_NAME, "pooled", ids=common.tolist(), strict=False)

    eff = ec.effective_dim(Z, var=0.95, ids=common)
    ec.write_manifest(
        JOINT_NAME,
        source=f"concat(zscore({vis}), zscore({aud}))",
        revision="joint-v1",
        weights_checksum={"applicable": False,
                          "reason": "cache concatenation; weights live in the source models"},
        n_params=None,
        modality="audiovisual",
        description=f"Joint: {vis} (visual) ++ {aud} (audio), each z-scored on train only.",
        concatenation_order=["visual:" + vis, "audio:" + aud],
        scaling="per-column z-score, mean/std fitted on train videos (id<=1000) only",
        visual_dim=int(Xv.shape[1]), audio_dim=int(Xa.shape[1]), joint_dim=int(Z.shape[1]),
        n_videos=int(len(common)), device="cpu", dtype="float32",
        wall_clock_s=round(time.perf_counter() - t0, 2),
        effective_dim=eff,
        silent_ids=K.SILENT_IDS, silent_ids_test=K.SILENT_IDS_TEST, n_silent=len(K.SILENT_IDS),
        mps_cpu_check={"applicable": False,
                       "reason": "pure numpy concatenation of cached arrays"},
        train_fit={"visual": sv, "audio": sa},
    )
    log_decision(
        agent="V-B", tags=["phase2", "joint", "concatenation", "scaling"],
        title="Joint audio-visual concatenation order and scaling",
        choice=f"joint_av = [z({vis}) | z({aud})], visual columns first then audio; each "
               f"column z-scored using mean/std from training videos (id<=1000) only. "
               f"Dims: visual {Xv.shape[1]} + audio {Xa.shape[1]} = {Z.shape[1]}, "
               f"over {len(common)} videos.",
        alternatives=[
            "unit-norm each modality instead of z-score: keeps relative scale but a "
            "high-variance modality still dominates the concatenation",
            "z-score on all 1102 videos: leaks test statistics into the fitted scaler",
        ],
        rationale="Per-column z-score on train stats equalises the two modalities' "
                  "scales without touching test data (constraint). Visual-first ordering "
                  "is recorded so downstream slicing is unambiguous.",
        consequences=f"If a source changes, joint_av must be rebuilt. Chosen sources were "
                     f"the highest-priority available: visual={vis}, audio={aud}.",
    )
    if verbose:
        print(f"joint_av built: visual={vis} ({Xv.shape[1]}d) + audio={aud} ({Xa.shape[1]}d) "
              f"= {Z.shape[1]}d over {len(common)} videos; {cons['path']}")
    return {"built": True, "visual": vis, "audio": aud, "shape": list(Z.shape),
            "n_videos": int(len(common)), "effective_dim": eff}
