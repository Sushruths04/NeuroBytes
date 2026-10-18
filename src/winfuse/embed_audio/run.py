"""Agent V-B extraction driver.

    export HF_HOME="$PWD/cache/hf" PYTORCH_ENABLE_MPS_FALLBACK=1
    .venv-embed/bin/python -m winfuse.embed_audio.run --model all [--limit 20] [--device mps|cpu]

For each neural model it (1) runs verify_device_equivalence on 20 videos before the
full run and records the verdict (falling back to CPU on failure), (2) extracts every
pending video resume-safely into cache/embeddings/<model>/_parts, (3) consolidates all
variants, (4) writes a manifest with provenance + silent_ids + effective_dim, and
(5) writes results/<model>/metrics.json via ResultsWriter.

Handcrafted descriptors skip the device check (CPU numpy/librosa/cv2) and say so.

`--model all` additionally rebuilds joint_av at the end from whatever caches exist.
"""
from __future__ import annotations

import argparse
import gc
import time

import numpy as np

from winfuse.common import embed_cache as ec
from winfuse.common.config import SEED
from winfuse.common.decisions import log_decision
from winfuse.common.embed_api import (BaseEmbedder, MpsFallbackRecorder, configure_torch,
                                   get_embedder, list_embedders, resolve_device,
                                   verify_device_equivalence)
from winfuse.common.frames import sampling_spec
from winfuse.common.io import test_video_path, train_video_path
from winfuse.common.results import ResultsWriter

from . import constants as K
from .joint import JOINT_NAME, build_joint

ALL_IDS = list(range(1, 1001)) + list(range(1001, 1103))   # 1..1102
HANDCRAFTED = ["hc_audio", "hc_visual", "hc_av"]
NEURAL = ["wav2vec2_base", "whisper_tiny", "whisper_base", "clap_audio", "ast_audioset"]
# default run set, in a sensible order (fast handcrafted first, then neural)
DEFAULT_MODELS = HANDCRAFTED + ["wav2vec2_base", "whisper_tiny", "whisper_base",
                                "clap_audio", "ast_audioset"]


def _video_path(vid: int) -> str:
    return str(train_video_path(vid) if vid <= 1000 else test_video_path(vid))


def _variants_of(result: dict) -> list[str]:
    """Which cache variants a model emits, from one sample embedding."""
    v = []
    if "pooled" in result:
        v.append("pooled")
    if "timeseries" in result:
        v.append("timeseries")
    for k in result.get("layers", {}):
        v.append(f"layer_{k}")
    return v


def extract_model(name: str, ids: list[int], device: str) -> dict:
    """Resume-safe extraction + consolidation + manifest + results for one model."""
    print(f"\n=== {name} on {device} ===", flush=True)
    is_handcrafted = name in HANDCRAFTED
    t_start = time.perf_counter()

    # probe one video to learn the variant set + times
    emb = get_embedder(name, device=device)
    sample = emb.embed(_video_path(ids[0]))
    variants = _variants_of(sample)
    times = sample.get("times")
    n_params = emb.n_params()

    # ---- device equivalence (neural only) ----
    mps_check = {"applicable": False,
                 "reason": "handcrafted CPU descriptor (numpy/librosa/cv2); no torch device"}
    if not is_handcrafted:
        probe_ids = ids[:20] if len(ids) >= 20 else ids
        paths = [_video_path(v) for v in probe_ids]
        mps_check = verify_device_equivalence(lambda: get_embedder(name), paths, key="pooled")
        if mps_check.get("skipped"):
            print(f"  device check skipped: {mps_check.get('reason')}")
        elif not mps_check.get("passed"):
            print(f"  device equivalence FAILED (cos_min={mps_check.get('cosine_min')}); "
                  f"falling back to CPU")
            log_decision(
                agent="V-B", tags=["phase2", "mps", "fallback", name],
                title=f"{name}: MPS/CPU equivalence failed, using CPU",
                choice=f"Fell back to CPU for {name}; per-video cosine min "
                       f"{mps_check.get('cosine_min')} < 0.99.",
                alternatives=["keep MPS and accept divergence (rejected: breaks the "
                              "numerical-equivalence gate for reported statistics)"],
                rationale="The equivalence gate is a correctness gate, not a permission gate.",
                consequences=f"{name} runs on CPU; slower but numerically matched.",
            )
            device = "cpu"
            emb = get_embedder(name, device=device)
        else:
            print(f"  device equivalence OK (cos_min={mps_check.get('cosine_min'):.5f}, "
                  f"speedup {mps_check.get('speedup_mps')})")

    # ---- extraction (resume-safe), capturing MPS fallbacks ----
    recorder = MpsFallbackRecorder()
    n_done = 0
    with recorder:
        # A video is pending if ANY of its variants is missing; we embed once and save
        # all variants, so each video is computed at most once.
        need = sorted({v for var in variants for v in ec.pending(name, var, ids)})
        print(f"  {len(need)} videos pending (of {len(ids)}) across {len(variants)} variants",
              flush=True)
        t0 = time.perf_counter()
        for k, vid in enumerate(need):
            r = emb.embed(_video_path(vid))
            if "pooled" in r:
                ec.save_part(name, "pooled", vid, r["pooled"])
            if "timeseries" in r:
                ec.save_part(name, "timeseries", vid, r["timeseries"])
            for lk, lv in r.get("layers", {}).items():
                ec.save_part(name, f"layer_{lk}", vid, lv)
            n_done += 1
            if (k + 1) % 50 == 0:
                print(f"    {k+1}/{len(need)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    # ---- consolidate every variant ----
    # Consolidate over the UNION of the requested ids and every part already on disk,
    # so a `--limit 20` smoke run can never shrink an existing full 1102 consolidation.
    cons = {}
    for variant in variants:
        tv = times if (variant == "timeseries") else None
        all_parts = sorted(set(ids) | set(ec.done_ids(name, variant)))
        cons[variant] = ec.consolidate(name, variant, ids=all_parts, times=tv, strict=False)

    wall = round(time.perf_counter() - t_start, 2)

    # ---- effective dim on pooled (train-only PCA) ----
    eff = None
    if "pooled" in variants:
        X, cids = ec.load(name, "pooled")
        eff = ec.effective_dim(X, var=0.95, ids=cids)

    # ---- weights checksum (neural) ----
    wsum = {}
    if not is_handcrafted:
        try:
            from huggingface_hub import try_to_load_from_cache  # noqa: F401
            import glob, os
            hf_home = os.environ.get("HF_HOME", "")
            repo_dir = name  # best-effort; checksum any cached safetensors/bin for the repo
            cands = glob.glob(os.path.join(hf_home, "**", "*.safetensors"), recursive=True)
            cands += glob.glob(os.path.join(hf_home, "**", "*.bin"), recursive=True)
            key = emb.source.split("/")[-1]
            cands = [c for c in cands if key.lower().replace("-", "") in c.lower().replace("-", "")]
            from pathlib import Path
            wsum = ec.weights_checksum([Path(c) for c in cands[:4]], limit_per_file=1 << 24)
        except Exception as e:  # noqa: BLE001
            wsum = {"error": repr(e)}

    silent_covered = [v for v in K.SILENT_IDS if v in set(ec.done_ids(name, variants[0]))]

    ec.write_manifest(
        name,
        source=emb.source, revision=emb.revision,
        weights_checksum=wsum, n_params=n_params,
        modality=emb.modality, description=emb.description,
        device=("cpu" if is_handcrafted else device), dtype="float32",
        frame_sampling=sampling_spec(),
        preprocessing=("16 kHz mono 3 s via src.common.frames.extract_audio"
                       if emb.modality in ("audio", "handcrafted") else "see frames"),
        wall_clock_s=wall, variants=variants,
        mps_cpu_check=mps_check,
        mps_fallback=recorder.summary(),
        silent_ids=K.SILENT_IDS,
        silent_ids_test=K.SILENT_IDS_TEST,
        n_silent=len(K.SILENT_IDS),
        silent_ids_covered_in_cache=len(silent_covered),
        effective_dim=eff,
        consolidation={v: cons[v]["shape"] for v in cons},
    )

    # ---- results ----
    rw = ResultsWriter(name, agent="V-B")
    pooled_shape = cons.get("pooled", {}).get("shape")
    rw.set_meta(
        description=emb.description,
        n_features=(pooled_shape[1] if pooled_shape else None),
        params={"source": emb.source, "revision": emb.revision,
                "variants": variants, "device": ("cpu" if is_handcrafted else device),
                "n_params": n_params},
        device=("cpu" if is_handcrafted else device),
    )
    rw.add_subject("stimuli", {
        "n_features": (pooled_shape[1] if pooled_shape else None),
        "runtime_s": {"transform": wall},
        "n_videos": (pooled_shape[0] if pooled_shape else None),
        "effective_dim": eff,
        "n_silent": len(K.SILENT_IDS),
        "mps_verdict": ("n/a (cpu handcrafted)" if is_handcrafted
                        else ("skipped" if mps_check.get("skipped")
                              else ("pass" if mps_check.get("passed") else "fail->cpu"))),
    })
    rw.add_note(f"Extraction facts: {len(ALL_IDS)} videos targeted, pooled cache "
                f"{pooled_shape}. Device {('cpu' if is_handcrafted else device)}. "
                f"Variants: {variants}.")
    rw.add_note(f"Audio coverage caveat: {len(K.SILENT_IDS)} of 1102 videos carry no usable "
                f"audio (34 no stream + 9 silent); they get a zero waveform and therefore a "
                f"constant/finite embedding that carries no audio information. "
                f"{len(K.SILENT_IDS_TEST)} are in the 102-video test pool "
                f"(ids {K.SILENT_IDS_TEST}), of which {K.NO_AUDIO_TEST_IDS} have no audio "
                f"stream at all. This dilutes any audio-based result.")
    rw.add_note("No V-C comparison metrics are written here; retrieval/RSA are V-C's.")
    jp, mp = rw.write()
    print(f"  wrote {jp}")

    del emb
    gc.collect()
    return {"model": name, "variants": variants, "pooled_shape": pooled_shape,
            "wall_s": wall, "device": ("cpu" if is_handcrafted else device),
            "effective_dim": eff, "mps_check": mps_check, "n_done": n_done}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="all",
                    help="model name, comma list, 'all', 'handcrafted', or 'neural'")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N video IDs (for the 20-video smoke run)")
    ap.add_argument("--device", default=None, help="mps|cpu (default: resolve_device())")
    args = ap.parse_args()

    env = configure_torch()
    if args.device:
        import os
        os.environ["EEG_EMBED_DEVICE"] = args.device
    device = resolve_device(args.device)
    print(f"torch env: {env}")
    print(f"device: {device}")

    np.random.seed(SEED)
    known = list_embedders()
    if args.model == "all":
        models = [m for m in DEFAULT_MODELS if m in known]
    elif args.model == "handcrafted":
        models = [m for m in HANDCRAFTED if m in known]
    elif args.model == "neural":
        models = [m for m in NEURAL if m in known]
    else:
        models = [m.strip() for m in args.model.split(",") if m.strip()]
    print(f"registered embedders: {known}")
    print(f"running: {models}")

    ids = ALL_IDS if not args.limit else ALL_IDS[:args.limit]

    rows = []
    for m in models:
        try:
            rows.append(extract_model(m, ids, device))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            rows.append({"model": m, "error": repr(e)})

    # joint, only on a full-ish run
    joint_status = None
    if args.model in ("all",) and not args.limit:
        joint_status = build_joint()
        print(f"joint: {joint_status}")

    # ---- final per-model table ----
    print("\n" + "=" * 92)
    print(f"{'model':<16}{'dim':>7}{'eff95':>8}{'wall_s':>9}{'device':>8}"
          f"{'mps':>10}{'n_silent':>10}")
    print("-" * 92)
    for r in rows:
        if "error" in r:
            print(f"{r['model']:<16}  ERROR: {r['error'][:60]}")
            continue
        ps = r.get("pooled_shape")
        dim = ps[1] if ps else "-"
        eff = r.get("effective_dim") or {}
        effv = eff.get("n_components_95pct", "-")
        mc = r.get("mps_check", {})
        mps = ("n/a" if not mc.get("applicable", True) and mc.get("reason", "").startswith("hand")
               else ("skip" if mc.get("skipped") else ("pass" if mc.get("passed") else "cpu")))
        if r["device"] == "cpu" and r["model"] in HANDCRAFTED:
            mps = "n/a"
        print(f"{r['model']:<16}{str(dim):>7}{str(effv):>8}{r['wall_s']:>9.1f}"
              f"{r['device']:>8}{mps:>10}{K.SILENT_IDS and len(K.SILENT_IDS):>10}")
    print("=" * 92)
    if joint_status:
        print(f"joint_av: {joint_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
