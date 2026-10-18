"""Agent V-A extraction driver.

    .venv-embed/bin/python -m winfuse.embed_visual.run --model all \
        [--limit 20] [--variants pooled,timeresolved] [--device mps|cpu]

For each model it:
  1. runs the MPS-vs-CPU equivalence gate on 20 videos (unless --device cpu),
     falling back to CPU + logging a decision if cosine_min < 0.99;
  2. extracts every requested variant resume-safely (per-video parts via embed_cache),
     wrapped in an MpsFallbackRecorder;
  3. consolidates parts into <variant>.npy + ids.npy, plus one array per `layers` key
     stored as variant `layer_<name>`;
  4. computes effective dimensionality (PCA 95% var, fitted on TRAIN videos only);
  5. writes cache/embeddings/<model>/manifest.json and results/<model>/{metrics,summary};
  6. prints a per-model row: dim / effective dim / wall-clock / device / MPS verdict.

Variants
--------
`pooled`       -> stored array (n, D)       (mean over 8 pooled-variant frames)
`timeresolved` -> stored array (n, 12, D)   (per-0.25-s frame)
`layers`       -> one array per layer, variant name `layer_<key>`, shape (n, Dk),
                  only when --variants includes "layers" (default: on for "all" runs).

The pooled variant drives the equivalence gate and effective-dim report. Only the 102
test VIDEOS are embedded from the test set; no EEG and no labels are ever read here.
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np

from winfuse.common import embed_cache as ec
from winfuse.common import io as cio
from winfuse.common.decisions import log_decision
from winfuse.common.embed_api import (MpsFallbackRecorder, configure_torch, embedder_class,
                                   get_embedder, resolve_device, verify_device_equivalence)
from winfuse.common.results import ResultsWriter

VISUAL_MODELS = ["alexnet_layers", "resnet50", "clip_vit_b32",
                 "dinov2_small", "dinov2_base", "videomae_base", "xclip_base"]

TRAIN_IDS = list(range(1, 1001))
TEST_IDS = list(range(1001, 1103))
ALL_IDS = TRAIN_IDS + TEST_IDS


def _vid_path(vid: int) -> str:
    return str(cio.train_video_path(vid) if vid <= 1000 else cio.test_video_path(vid))


def _part_variants(model: str, want_variants: list[str]) -> list[str]:
    """Resolve requested logical variants into concrete per-video part variants."""
    out = []
    if "pooled" in want_variants:
        out.append(("pooled", "pooled"))
    if "timeresolved" in want_variants:
        out.append(("timeresolved", "timeresolved"))
    return out


def _extract(model: str, ids: list[int], want_variants: list[str], device: str,
             want_layers: bool) -> dict:
    """Resume-safe extraction. Returns timing + fallback info."""
    emb = get_embedder(model, device=device)

    # discover which part-variants to write per video
    pooled_on = "pooled" in want_variants
    tr_on = "timeresolved" in want_variants

    # pending ids: a video is "done" only if all requested top-level variants exist
    def pending_ids() -> list[int]:
        todo = set()
        if pooled_on:
            todo |= set(ec.pending(model, "pooled", ids))
        if tr_on:
            todo |= set(ec.pending(model, "timeresolved", ids))
        if want_layers:
            # need at least the first layer part to detect done; probe layer set lazily
            pass
        return sorted(todo)

    todo = pending_ids()
    layer_variants: set[str] = set()
    t0 = time.perf_counter()
    rec = MpsFallbackRecorder()
    n_sub_total = 0
    with rec:
        for k, vid in enumerate(todo):
            out = emb.embed(_vid_path(vid))
            if pooled_on and "pooled" in out:
                ec.save_part(model, "pooled", vid, out["pooled"])
            if tr_on and "timeseries" in out:
                ec.save_part(model, "timeresolved", vid, out["timeseries"])
            if want_layers and "layers" in out:
                for lname, lvec in out["layers"].items():
                    lv = f"layer_{lname}"
                    layer_variants.add(lv)
                    ec.save_part(model, lv, vid, lvec)
            if (k + 1) % 25 == 0 or (k + 1) == len(todo):
                print(f"    {model}: {k+1}/{len(todo)} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    wall = time.perf_counter() - t0

    # if nothing was pending, still recover the layer variant names from disk
    if want_layers and not layer_variants:
        d = ec.model_dir(model) / "_parts"
        if d.exists():
            layer_variants = {p.name for p in d.iterdir()
                              if p.is_dir() and p.name.startswith("layer_")}

    return {"wall_clock_s": wall, "n_extracted": len(todo),
            "layer_variants": sorted(layer_variants),
            "fallback": rec.summary(), "fell_back": rec.fell_back}


def _consolidate_all(model: str, ids: list[int], want_variants: list[str],
                     layer_variants: list[str]) -> dict:
    from winfuse.common.frames import frame_times

    info = {}
    if "pooled" in want_variants:
        info["pooled"] = ec.consolidate(model, "pooled", ids, strict=True)
    if "timeresolved" in want_variants:
        info["timeresolved"] = ec.consolidate(
            model, "timeresolved", ids, times=frame_times("timeresolved"), strict=True)
    for lv in layer_variants:
        info[lv] = ec.consolidate(model, lv, ids, strict=True)
    return info


def _effective_dim(model: str) -> dict | None:
    try:
        X, ids = ec.load(model, "pooled")
        return ec.effective_dim(X, var=0.95, ids=ids)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def _weights_checksum(model: str) -> dict:
    """Checksum the on-disk weight files for provenance (best-effort)."""
    cls = embedder_class(model)
    src = getattr(cls, "source", "")
    paths: list[Path] = []
    # torchvision checkpoints live in ~/.cache/torch/hub/checkpoints
    if src.startswith("torchvision"):
        hub = Path.home() / ".cache/torch/hub/checkpoints"
        if hub.exists():
            paths = sorted(hub.glob("*.pth"))
    else:
        # HF snapshot dir under HF_HOME
        import os
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        repo = "models--" + src.replace("/", "--")
        snap = hf_home / "hub" / repo / "snapshots"
        if snap.exists():
            for d in snap.iterdir():
                paths += list(d.glob("*.safetensors")) + list(d.glob("*.bin"))
    return ec.weights_checksum(paths) if paths else {}


def run_model(model: str, ids: list[int], want_variants: list[str], device_arg: str | None,
              want_layers: bool, eq_videos: int = 20) -> dict:
    print(f"\n=== {model} ===", flush=True)
    cls = embedder_class(model)
    device = device_arg or resolve_device()

    # 1. device equivalence gate (only when targeting mps)
    eq = {"skipped": True, "reason": "device=cpu requested"}
    if device == "mps":
        eq_paths = [_vid_path(v) for v in ids[:eq_videos]]
        eq = verify_device_equivalence(lambda: get_embedder(model), eq_paths, key="pooled")
        if not eq.get("passed"):
            log_decision(
                agent="V-A", tags=["phase2", "device", model],
                title=f"{model}: MPS/CPU equivalence failed -> CPU",
                choice=f"Fell back to CPU for {model}. cosine_min="
                       f"{eq.get('cosine_min')}, required 0.99.",
                alternatives=["Keep MPS: rejected, would make reported embeddings "
                              "device-dependent beyond the 0.99 gate."],
                rationale="The frozen contract requires cosine>=0.99 MPS-vs-CPU before a "
                          "full run; this model did not meet it.",
                consequences="Slower extraction for this model; embeddings are CPU-exact.")
            device = "cpu"
        else:
            print(f"  MPS equivalence: cosine_min={eq.get('cosine_min'):.5f} "
                  f"(speedup {eq.get('speedup_mps')})", flush=True)

    # 2. extract
    ext = _extract(model, ids, want_variants, device, want_layers)
    if ext["fell_back"]:
        log_decision(
            agent="V-A", tags=["phase2", "mps_fallback", model],
            title=f"{model}: MPS ops fell back to CPU",
            choice=f"{ext['fallback']['n_fallback_warnings']} fallback warning(s) during "
                   f"{model} extraction on {device}.",
            alternatives=["Force all ops to a device: not possible without reimplementing "
                          "the model."],
            rationale="PYTORCH_ENABLE_MPS_FALLBACK=1 routes unsupported ops to CPU "
                      "transparently; recorded for provenance.",
            consequences="Mixed-device execution; the 0.99 equivalence gate still bounds "
                         "any numerical effect.",
            allow_duplicate=False)

    # 3. consolidate
    cons = _consolidate_all(model, ids, want_variants, ext["layer_variants"])

    # 4. effective dim (train-only PCA)
    effd = _effective_dim(model)

    # 5. manifest
    try:
        emb_for_params = get_embedder(model, device="cpu")
        emb_for_params._load()
        emb_for_params._loaded = True
        n_params = emb_for_params.n_params()
    except Exception:
        n_params = None

    pooled_shape = cons.get("pooled", {}).get("shape")
    dim = pooled_shape[-1] if pooled_shape else None
    ec.write_manifest(
        model,
        source=cls.source, revision=cls.revision,
        weights_checksum=_weights_checksum(model),
        n_params=n_params, device=device, dtype="float32",
        modality=cls.modality, description=cls.description, agent="V-A",
        frame_sampling={"pooled": "8 frames, bin centres 0.1875..2.8125 s",
                        "timeresolved": "12 frames, 0.25 s grid 0.0..2.75 s",
                        "variant_source": "winfuse.common.frames (frozen)"},
        preprocessing=f"{model} native processor; 268->224 downsample (bilinear/bicubic)",
        wall_clock_s=round(ext["wall_clock_s"], 1),
        mps_cpu_check=eq,
        mps_fallback=ext["fallback"],
        variants=list(cons.keys()),
        effective_dim=effd,
        n_videos=cons.get("pooled", {}).get("n_videos"),
        pooled_dim=dim,
    )

    # 6. results
    rw = ResultsWriter(model, agent="V-A")
    eff_key = next((k for k in (effd or {}) if k.startswith("n_components_")), None)
    rw.set_meta(
        description=cls.description,
        n_features=int(dim) if dim else None,
        params={"source": cls.source, "revision": cls.revision,
                "n_params": n_params, "device": device,
                "variants": list(cons.keys()),
                "n_layers": len(ext["layer_variants"])},
    )
    extraction = {
        "pooled_dim": dim,
        "effective_dim_95pct": (effd.get(eff_key) if eff_key else None),
        "ambient_dim": (effd or {}).get("ambient_dim"),
        "participation_ratio": (effd or {}).get("participation_ratio"),
        "n_params": n_params,
        "device": device,
        "wall_clock_s": round(ext["wall_clock_s"], 1),
        "n_videos": cons.get("pooled", {}).get("n_videos"),
        "mps_cpu_check": {"passed": eq.get("passed"), "cosine_min": eq.get("cosine_min"),
                          "skipped": eq.get("skipped", False),
                          "reason": eq.get("reason")},
        "mps_fallback_warnings": ext["fallback"]["n_fallback_warnings"],
    }
    rw.add_subject("stimulus_set", extraction)
    rw.add_note("Extraction facts only. Agent V-C computes EEG-vs-embedding comparison "
                "metrics; those keys are intentionally absent here.")
    rw.add_note(f"effective_dim is PCA components for 95% variance fitted on TRAIN videos "
                f"only (1..1000): {effd}")
    if ext["fallback"]["n_fallback_warnings"]:
        rw.add_note(f"MPS fallback ops: {ext['fallback']['mps_fallback_ops']}")
    rw.write()

    row = {
        "model": model, "dim": dim,
        "eff_dim": (effd.get(eff_key) if eff_key else None),
        "wall_s": round(ext["wall_clock_s"], 1), "device": device,
        "cos_min": eq.get("cosine_min"), "eq_skipped": eq.get("skipped", False),
        "n_videos": cons.get("pooled", {}).get("n_videos"),
        "n_layers": len(ext["layer_variants"]),
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="all",
                    help="comma list of model names, or 'all'")
    ap.add_argument("--limit", type=int, default=None,
                    help="embed only the first N video IDs (for a smoke run)")
    ap.add_argument("--variants", default="pooled,timeresolved")
    ap.add_argument("--device", default=None, choices=[None, "mps", "cpu"])
    ap.add_argument("--no-layers", action="store_true",
                    help="skip per-layer caching (pooled/timeresolved only)")
    args = ap.parse_args()

    envinfo = configure_torch()
    print(f"torch env: {envinfo}")

    models = VISUAL_MODELS if args.model == "all" else [m.strip() for m in args.model.split(",")]
    want_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    want_layers = not args.no_layers

    ids = ALL_IDS if args.limit is None else ALL_IDS[:args.limit]
    print(f"videos: {len(ids)} (ids {ids[0]}..{ids[-1]}), variants={want_variants}, "
          f"layers={want_layers}, device={args.device or 'auto'}")

    rows = []
    for m in models:
        try:
            rows.append(run_model(m, ids, want_variants, args.device, want_layers))
        except Exception as e:  # noqa: BLE001
            print(f"  !! {m} FAILED: {e}")
            traceback.print_exc()
            rows.append({"model": m, "dim": None, "eff_dim": None, "wall_s": None,
                         "device": args.device, "cos_min": None, "error": repr(e)[:80]})

    # per-model table
    print("\n" + "=" * 92)
    print(f"{'model':<16}{'dim':>7}{'eff_dim':>9}{'wall_s':>9}{'device':>8}"
          f"{'cos_min':>10}{'n_vid':>8}{'layers':>8}")
    print("-" * 92)
    for r in rows:
        cm = r.get("cos_min")
        cm_s = "skip" if r.get("eq_skipped") else (f"{cm:.5f}" if cm is not None else "–")
        print(f"{r['model']:<16}{str(r.get('dim') or '–'):>7}"
              f"{str(r.get('eff_dim') or '–'):>9}{str(r.get('wall_s') or '–'):>9}"
              f"{str(r.get('device') or '–'):>8}{cm_s:>10}"
              f"{str(r.get('n_videos') or '–'):>8}{str(r.get('n_layers') or '–'):>8}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
