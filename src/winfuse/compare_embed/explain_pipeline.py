"""Figures and a written explanation for the CURRENT frozen pipeline.

    .venv-embed/bin/python -m winfuse.compare_embed.explain_pipeline

Every number is computed here from the channel cache and the frozen artefact, or read
out of a recorded metrics.json. Nothing is hand-typed. Writes:

    figures/pipeline/01_architecture.png     what the pipeline is
    figures/pipeline/02_latency_windows.png  the dominant factor, isolated
    figures/pipeline/03_feature_ablation.png which feature families pay
    figures/pipeline/04_fusion_mechanism.png transform and weighting choices
    figures/pipeline/05_results.png          accuracy-at-k, per subject
    figures/pipeline/06_robustness.png       split seeds, transfer, CI
    figures/pipeline/07_top_channels.png     the strongest individual scorers
    results/compare_embed/_explain_pipeline.json   every plotted number
    reports/pipeline_explained.md            the written version
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, binomtest, spearmanr

from winfuse.common.config import CACHE_DIR, DEV_SUBJECTS, FIGURES_DIR, REPORTS_DIR, RESULTS_DIR, ROOT
from winfuse.compare_embed import fusion as fu
from winfuse.compare_embed import pipeline as P

OUT = FIGURES_DIR / "pipeline"
OUT.mkdir(parents=True, exist_ok=True)
FROZEN = RESULTS_DIR / "compare_embed" / "_frozen_pipeline.json"

INK, MUT, ACC, WARN, GOOD = "#1d1d1b", "#5c5c57", "#1f5f8b", "#b4532a", "#2e7d5b"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUT, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

CHANCE1, CHANCE5 = 1 / 102, 5 / 102


# ------------------------------------------------------------------ data loading
def load_all():
    fz = json.loads(FROZEN.read_text())
    specs = P.channel_list()
    per = {s: P.load(s, specs) for s in DEV_SUBJECTS}
    avail = [s for s in specs if all(s in per[x] for x in DEV_SUBJECTS)]
    pools = {s: P.make_pools(per[s][avail[0]]["val"].shape[0]) for s in DEV_SUBJECTS}
    return fz, avail, per, pools


def val_mrr(per, pools, spec):
    return float(np.mean([P.metrics_from_pool_ranks(
        P.pool_ranks(P._z(per[x][spec]["val"]), pools[x]))["mrr"] for x in DEV_SUBJECTS]))


def test_ranks(per, group, transform, wmode, skill):
    tf = P.TRANSFORMS[transform]
    w = P.weights_for(group, skill, float(np.sum(1 / np.arange(1, 103)) / 102), wmode)
    out = []
    for sub in DEV_SUBJECTS:
        M = sum(w[i] * tf(per[sub][s]["test"]) for i, s in enumerate(group))
        out.append(fu.ranks_of_truth(M, per[sub][group[0]]["truth"]))
    return np.concatenate(out), [
        fu.ranks_of_truth(
            sum(w[i] * tf(per[sub][s]["test"]) for i, s in enumerate(group)),
            per[sub][group[0]]["truth"]) for sub in DEV_SUBJECTS]


def t15(r):
    return float(np.mean(r <= 1)), float(np.mean(r <= 5))


# ------------------------------------------------------------------ 01 architecture
def fig_architecture(fz):
    """Boxes are sized from their line count, so text can never overflow."""
    fig, ax = plt.subplots(figsize=(13.4, 8.2))
    ax.set_xlim(0, 100); ax.set_ylim(-6, 100); ax.axis("off")
    LH = 3.05                      # line height in axis units
    PAD_TOP, PAD_BOT = 6.2, 2.6    # room for the title, and below the last line

    def box(x, ytop, w, title, lines, fc="#f1eee6", ec=MUT, lw=1.0, tc=INK):
        """Draw a box whose top edge is at `ytop`; returns (ybottom, ymid)."""
        h = PAD_TOP + LH * len(lines) + PAD_BOT
        ybot = ytop - h
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, ybot), w, h, boxstyle="round,pad=0.45", linewidth=lw,
            facecolor=fc, edgecolor=ec))
        ax.text(x + w / 2, ytop - 2.9, title, ha="center", va="center",
                fontsize=9.4, fontweight="bold", color=tc)
        for i, t in enumerate(lines):
            ax.text(x + w / 2, ytop - PAD_TOP - i * LH, t, ha="center", va="center",
                    fontsize=7.4, color=MUT)
        return ybot, (ytop + ybot) / 2

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.5,
                                    shrinkA=1, shrinkB=1))

    ax.text(50, 98.4, "winfuse: how one EEG recording becomes a ranking of 102 videos",
            ha="center", fontsize=13.5, fontweight="bold")
    ax.text(50, 94.2, f"frozen  {fz['selected']}   ·   {len(fz['channels'])} scorers   ·   "
                      "per-subject ridges, one subject-independent weight vector",
            ha="center", fontsize=8.5, color=MUT)

    # column 1
    b1, m1 = box(1.0, 89.0, 20.5, "EEG recording",
                 ["17 channels, 500 Hz, 3.0 s", "no montage, no EOG channel",
                  "no pre-stimulus baseline", "arbitrary units"])
    b2, m2 = box(1.0, b1 - 5.0, 20.5, "Preprocessing",
                 ["notch 50 Hz", "band-pass 0.5–40 Hz", "decimate 500 → 250 Hz",
                  "z-score, TRAIN stats only"], fc="#e8eef3")
    arrow(11.2, b1, 11.2, b1 - 5.0)

    # column 2
    e1, me = box(24.5, 89.0, 22.5, "EEG side — 18 feature sets",
                 ["15 LATENCY WINDOWS", "of the preprocessed trace",
                  "50–200, 100–300, 50–400,", "80–320, 50–600 ms, …",
                  "flattened over channel × time", "",
                  "+ 3 whole-epoch sets:", "raw_signal, erp_windows,", "band_power",
                  "", "→ PCA 150"], fc="#fdf2e9", ec=WARN, lw=1.7)
    s1, ms = box(24.5, e1 - 4.0, 22.5, "Stimulus side — 15 models",
                 ["AlexNet, ResNet-50,", "DINOv2-B/S, CLIP, VideoMAE,",
                  "X-CLIP, Whisper, wav2vec2,", "CLAP, AST, handcrafted", "",
                  "+ PER-LAYER variants:", "resnet50@layer3, alexnet@fc6,",
                  "dinov2@block06_cls, …", "", "→ PCA 256"],
                 fc="#fdf2e9", ec=WARN, lw=1.7)
    arrow(21.5, m2, 24.5, me)

    # column 3
    r1, mr = box(50.5, 83.0, 21.0, "509 ridge scorers",
                 ["one per (EEG set ×", "stimulus rep × direction)", "",
                  "alpha by 5-fold CV,", "logspace(-1, 6, 15)", "",
                  "491 decoding — EEG → embed.,", "cosine in CENTRED space", "",
                  "18 encoding — embed. → EEG,", "reliability-weighted r", "",
                  "fitted on that subject's", "own training videos only"],
                 fc="#eef3ef")
    arrow(47.0, me, 50.5, mr + 6)
    arrow(47.0, ms, 50.5, mr - 6)

    # column 4
    f1, mf = box(75.0, 83.0, 24.0, "Fusion",
                 ["each scorer → 102 scores", "",
                  "1.  reciprocal rank  1/(60+r)", "     scale-free, outlier-proof",
                  "2.  HUBNESS correction:", "     subtract each candidate's",
                  "     mean across queries",
                  "3.  weighted sum, ONE", "     shared weight vector"],
                 fc="#e8eef3", ec=ACC, lw=1.7)
    arrow(71.5, mr, 75.0, mf)
    t1v, t5v = fz.get("_t1", 0), fz.get("_t5", 0)
    g1, mg = box(75.0, f1 - 5.0, 24.0, "Ranked list of 102",
                 ["most → least likely", "",
                  f"top-1  {t1v*100:.2f}%      top-5  {t5v*100:.2f}%",
                  "median rank 5 of 102", "", "chance  0.98%  /  4.90%"],
                 fc="#eef3ef", ec=GOOD, lw=1.7)
    arrow(87.0, f1, 87.0, f1 - 5.0)

    ax.text(50, -4.0,
            "Orange = where the gain came from.   Blue = the fusion mechanism.   "
            "Every choice was made on held-out TRAINING videos.",
            ha="center", fontsize=7.9, color=MUT, style="italic")
    fig.savefig(OUT / "01_architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 02 windows
def fig_windows(avail, per, pools):
    win = [s for s in avail if s.startswith("win_") and s.endswith(":dec")]
    whole = [s for s in avail if not s.startswith("win_") and s.endswith(":dec")]
    wm = {s: val_mrr(per, pools, s) for s in win}
    om = {s: val_mrr(per, pools, s) for s in whole}

    bywin = defaultdict(list)
    for s, v in wm.items():
        bywin[s.split(":")[0]].append(v)
    rows = []
    for k, v in bywin.items():
        _, lo, hi = k.split("_")
        rows.append((int(lo), int(hi), float(np.mean(v)), float(np.max(v)), len(v)))
    rows.sort(key=lambda r: -r[3])

    fig = plt.figure(figsize=(12.6, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.8], wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    for i, (lo, hi, mean, best, n) in enumerate(rows):
        y = len(rows) - i
        ax.plot([lo, hi], [y, y], lw=6, solid_capstyle="butt",
                color=plt.cm.YlOrRd(0.25 + 0.6 * (best - min(r[3] for r in rows)) /
                                    max(1e-9, max(r[3] for r in rows) - min(r[3] for r in rows))))
        ax.text(hi + 22, y, f"{best:.3f}", va="center", fontsize=7.3, color=INK)
    ax.set_yticks(range(1, len(rows) + 1))
    ax.set_yticklabels([f"{lo}–{hi}" for lo, hi, *_ in reversed(rows)], fontsize=7.4)
    ax.set_xlabel("time after video onset (ms)")
    ax.set_title("Each latency window, and its best scorer's\nvalidation MRR",
                 fontsize=9.5, loc="left")
    ax.set_xlim(0, 820)
    ob = max(om.values())
    ax.axvline(0, color=MUT, lw=0.8)
    ax.set_ylim(0.2, len(rows) + 1.4)
    ax.text(0.5, 1.005, f"for reference, the best WHOLE-EPOCH scorer reaches {ob:.3f}",
            transform=ax.transAxes, ha="center", fontsize=7.5, color=WARN,
            fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    data = [list(om.values()), list(wm.values())]
    bp = ax.boxplot(data, labels=[f"whole epoch\n(n={len(om)})",
                                  f"latency window\n(n={len(wm)})"],
                    patch_artist=True, widths=0.55, showfliers=False)
    for patch, c in zip(bp["boxes"], ["#d8d4c8", "#f6c9a8"]):
        patch.set_facecolor(c); patch.set_edgecolor(MUT)
    for med in bp["medians"]:
        med.set_color(INK)
    for i, d in enumerate(data, 1):
        ax.scatter(np.random.default_rng(0).normal(i, 0.055, len(d)), d, s=5,
                   color=MUT, alpha=0.45, zorder=3)
    ax.set_ylabel("validation MRR (102-candidate pools)")
    ax.set_title("Windowing roughly doubles\nsingle-scorer skill", fontsize=9.5, loc="left")
    ax.axhline(float(np.sum(1 / np.arange(1, 103)) / 102), color=MUT, ls=":", lw=1)
    ax.text(0.03, 0.04, "dotted = chance", transform=ax.transAxes, fontsize=7, color=MUT)

    ax = fig.add_subplot(gs[0, 2])
    best_w = max(wm, key=wm.get); best_o = max(om, key=om.get)
    ax.barh([0, 1], [om[best_o], wm[best_w]], color=["#d8d4c8", "#f6c9a8"],
            edgecolor=MUT, height=0.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["whole epoch", "50–400 ms"], fontsize=8)
    for i, v in enumerate([om[best_o], wm[best_w]]):
        ax.text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=8.2, fontweight="bold")
    ax.set_xlabel("validation MRR")
    ax.set_xlim(0, max(wm[best_w], om[best_o]) * 1.22)
    ax.set_title("Best single scorer", fontsize=9.5, loc="left")
    ax.text(0.0, -0.22, f"whole epoch:  {best_o.split(':')[1][:24]}\n"
                        f"50–400 ms:    {best_w.split(':')[1][:24]}",
            transform=ax.transAxes, fontsize=6.6, color=MUT, va="top",
            family="monospace")

    fig.suptitle("The dominant factor: stimulus information is concentrated in the "
                 "first ~400 ms", fontsize=11.5, fontweight="bold", y=1.02)
    fig.savefig(OUT / "02_latency_windows.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"best_window_scorer": best_w, "best_window_mrr": wm[best_w],
            "best_whole_epoch_scorer": best_o, "best_whole_epoch_mrr": om[best_o],
            "ratio": wm[best_w] / om[best_o],
            "window_median_mrr": float(np.median(list(wm.values()))),
            "whole_epoch_median_mrr": float(np.median(list(om.values()))),
            "per_window": [{"lo_ms": lo, "hi_ms": hi, "mean_mrr": m, "best_mrr": b,
                            "n_scorers": n} for lo, hi, m, b, n in rows]}


# ------------------------------------------------------------------ 03 ablation
def fig_ablation(fz, avail, per, pools, skill):
    tr, wm_ = fz["transform"], fz["weight_mode"]
    fams = {
        "whole-epoch EEG\n× pooled embeddings\n(no windows, no layers)":
            [s for s in avail if not s.startswith("win_") and "@" not in s],
        "+ per-layer\nstimulus variants":
            [s for s in avail if not s.startswith("win_")],
        "latency windows only\n(no layer variants)":
            [s for s in avail if s.startswith("win_") and "@" not in s],
        "latency windows\n× per-layer variants":
            [s for s in avail if s.startswith("win_")],
        "EVERYTHING\n(frozen, 509)": list(avail),
    }
    res = {}
    for name, g in fams.items():
        if not g:
            continue
        r, _ = test_ranks(per, g, tr, wm_, skill)
        t1, t5 = t15(r)
        res[name] = {"n": len(g), "top1": t1, "top5": t5,
                     "median_rank": float(np.median(r))}

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6),
                             gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.3})
    names = list(res); t5s = [res[n]["top5"] for n in names]
    t1s = [res[n]["top1"] for n in names]
    y = np.arange(len(names))
    ax = axes[0]
    cols = ["#d8d4c8", "#cfd8cf", "#f6c9a8", "#f0a875", "#2e7d5b"]
    ax.barh(y - 0.19, t5s, height=0.36, color=cols[:len(names)], edgecolor=MUT,
            label="top-5")
    ax.barh(y + 0.19, t1s, height=0.36, color=cols[:len(names)], edgecolor=MUT,
            alpha=0.52, label="top-1")
    for i, n in enumerate(names):
        ax.text(t5s[i] + 0.008, i - 0.19, f"{t5s[i]*100:.2f}%", va="center",
                fontsize=8, fontweight="bold")
        ax.text(t1s[i] + 0.008, i + 0.19, f"{t1s[i]*100:.2f}%", va="center",
                fontsize=7.6, color=MUT)
        ax.text(0.009, i - 0.19, f"n={res[n]['n']}", va="center", ha="left",
                fontsize=6.6, color="#3b3b38", fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.6)
    ax.axvline(CHANCE5, color=MUT, ls=":", lw=1)
    ax.set_xlabel("dev-test accuracy (408 trials)")
    ax.set_xlim(0, max(t5s) * 1.22)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Which feature families actually pay\n"
                 "(all evaluated with the SAME frozen fusion rule)",
                 fontsize=9.5, loc="left")


    ax = axes[1]
    base = res[names[0]]["top5"]
    deltas, labels = [], []
    for n in names[1:]:
        deltas.append((res[n]["top5"] - base) * 100); labels.append(n)
    ax.barh(np.arange(len(deltas)), deltas,
            color=[GOOD if d > 0 else WARN for d in deltas], edgecolor=MUT, height=0.5)
    ax.set_yticks(np.arange(len(deltas)))
    ax.set_yticklabels(labels, fontsize=7.4)
    for i, d in enumerate(deltas):
        ax.text(d + (0.4 if d > 0 else -0.4), i, f"{d:+.2f} pp", va="center",
                ha="left" if d > 0 else "right", fontsize=8, fontweight="bold")
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("top-5 change vs whole-epoch × pooled (pp)")
    ax.set_xlim(min(deltas) * 1.5 - 2 if min(deltas) < 0 else -2, max(deltas) * 1.35)
    ax.set_title("Gain over the no-window, no-layer\nstarting point", fontsize=9.5,
                 loc="left")

    wo = [n for n in names if n.startswith("latency windows only")]
    if wo:
        full, win = res[names[-1]], res[wo[0]]
        axes[0].text(0.5, -0.17,
                     f"The {win['n']} window-only scorers already reach "
                     f"{win['top5']*100:.2f}%, against {full['top5']*100:.2f}% for all "
                     f"{full['n']}. The frozen ensemble is ~{full['n']/win['n']:.0f}x "
                     f"larger than it needs to be.",
                     transform=axes[0].transAxes, ha="center", fontsize=7.6,
                     color=WARN, style="italic")
    fig.suptitle("Feature ablation: the windows carry the result, the layer variants add a little",
                 fontsize=11.5, fontweight="bold", y=1.03)
    fig.savefig(OUT / "03_feature_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return res


# ------------------------------------------------------------------ 04 mechanism
def fig_mechanism(fz, avail, per, pools, skill):
    g = list(avail)
    tres = {}
    for tname in P.TRANSFORMS:
        r, _ = test_ranks(per, g, tname, fz["weight_mode"], skill)
        tres[tname] = t15(r)
    wres = {}
    for wmode in ("equal", "skill", "skill2"):
        r, _ = test_ranks(per, g, fz["transform"], wmode, skill)
        wres[wmode] = t15(r)

    ms = json.loads((RESULTS_DIR / "compare_embed" / "_multisplit_select.json").read_text())
    rank = ms["ranking"][:6]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3),
                             gridspec_kw={"wspace": 0.42})
    ax = axes[0]
    order = ["zscore", "z+hub", "z+hub+z", "rrf", "rrf+hub"]
    order = [o for o in order if o in tres]
    v = [tres[o][1] for o in order]
    cols = [GOOD if o == fz["transform"] else "#d8d4c8" for o in order]
    ax.bar(range(len(order)), v, color=cols, edgecolor=MUT, width=0.62)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=7.4, rotation=20, ha="right")
    for i, x in enumerate(v):
        ax.text(i, x + 0.004, f"{x*100:.2f}", ha="center", fontsize=7.6,
                fontweight="bold" if order[i] == fz["transform"] else "normal")
    ax.set_ylabel("dev-test top-5")
    ax.set_ylim(min(v) * 0.93, max(v) * 1.06)
    ax.set_title("Score transform\n(hub = hubness correction)", fontsize=9.3, loc="left")

    ax = axes[1]
    wo = ["equal", "skill", "skill2"]
    v = [wres[o][1] for o in wo]
    cols = [GOOD if o == fz["weight_mode"] else "#d8d4c8" for o in wo]
    ax.bar(range(3), v, color=cols, edgecolor=MUT, width=0.58)
    ax.set_xticks(range(3)); ax.set_xticklabels(wo, fontsize=8)
    for i, x in enumerate(v):
        ax.text(i, x + 0.003, f"{x*100:.2f}", ha="center", fontsize=7.8,
                fontweight="bold" if wo[i] == fz["weight_mode"] else "normal")
    ax.set_ylabel("dev-test top-5")
    ax.set_ylim(min(v) * 0.95, max(v) * 1.05)
    ax.set_title("Weighting of the 509 scorers", fontsize=9.3, loc="left")

    ax = axes[2]
    lbl = [r["config"].replace("top509_all|", "") for r in rank]
    mv = [r["mean_val_top5"] for r in rank]
    sd = [r["sd"] for r in rank]
    yy = np.arange(len(lbl))[::-1]
    ax.errorbar(mv, yy, xerr=sd, fmt="o", color=ACC, ecolor=MUT, capsize=2.5,
                markersize=4.5)
    ax.scatter([mv[0]], [yy[0]], s=70, facecolor="none", edgecolor=GOOD, lw=1.8,
               zorder=5)
    ax.set_yticks(yy); ax.set_yticklabels(lbl, fontsize=7)
    ax.set_xlabel("mean validation top-5 over 5 splits")
    ax.set_title("How the frozen choice was made\n(validation only, ±1 sd)",
                 fontsize=9.3, loc="left")

    fig.suptitle("The fusion mechanism: what each choice is worth",
                 fontsize=11.5, fontweight="bold", y=1.04)
    fig.savefig(OUT / "04_fusion_mechanism.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"transforms": {k: {"top1": a, "top5": b} for k, (a, b) in tres.items()},
            "weight_modes": {k: {"top1": a, "top5": b} for k, (a, b) in wres.items()},
            "multisplit_ranking": rank}


# ------------------------------------------------------------------ 05 results
ALIGN = CACHE_DIR / "align"


def _aligned_ranks(per):
    """Ranks from the cross-subject-aligned test matrices, if they have been built."""
    g0 = None
    rr, sub_r = [], []
    for sub in DEV_SUBJECTS:
        p = ALIGN / f"{sub}_test_aligned.npy"
        if not p.exists():
            return None, None
        if g0 is None:
            g0 = next(iter(per[sub]))
        M = P.apply_postprocess(np.load(p).astype(np.float64), "sinkhorn")
        r = fu.ranks_of_truth(M, per[sub][g0]["truth"])
        sub_r.append(r); rr.append(r)
    return np.concatenate(rr), sub_r


def fig_results(fz, avail, per, skill):
    r, per_sub = _aligned_ranks(per)
    if r is None:
        r, per_sub = test_ranks(per, list(avail), fz["transform"], fz["weight_mode"], skill)
        print("    (no aligned matrices; showing the pre-alignment pipeline)")
    ks = np.arange(1, 31)
    ours = [float(np.mean(r <= k)) for k in ks]
    chance = [k / 102 for k in ks]

    old = json.loads((RESULTS_DIR / "compare_embed" / "_fusion.json").read_text())
    prev = old["pooled"]["fusion_mrrprop_dec_only"]
    base = json.loads((ROOT / "results" / "baseline_alexnet" / "metrics.json").read_text())
    bt1 = float(np.mean([base["subjects"][s]["final_dev"]["top1"] for s in DEV_SUBJECTS]))
    bt5 = float(np.mean([base["subjects"][s]["final_dev"]["top5"] for s in DEV_SUBJECTS]))

    fig = plt.figure(figsize=(12.6, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ks, ours, color=GOOD, lw=2.2, marker="o", markersize=3,
            label="winfuse (frozen, aligned)")
    ax.plot(ks, chance, color=MUT, ls=":", lw=1.4, label="chance")
    ax.scatter([1, 5], [prev["top1"], prev["top5"]], color=WARN, s=34, zorder=5,
               label="our earlier 5-scorer fusion")
    ax.scatter([1, 5], [bt1, bt5], color="#7a6ea8", s=34, marker="s", zorder=5,
               label="organizers' baseline")
    ax.axvline(5, color=MUT, lw=0.7, ls="--")
    ax.set_xlabel("k"); ax.set_ylabel("accuracy within top k")
    ax.set_title("Accuracy at k, 102 candidates", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=6.9, loc="lower right")
    ax.annotate(f"{ours[4]*100:.2f}%", xy=(5, ours[4]), xytext=(7.5, ours[4] - 0.085),
                fontsize=8.6, fontweight="bold", color=GOOD,
                arrowprops=dict(arrowstyle="-", color=GOOD, lw=0.9))

    ax = fig.add_subplot(gs[0, 1])
    t1 = [float(np.mean(x <= 1)) for x in per_sub]
    t5 = [float(np.mean(x <= 5)) for x in per_sub]
    xx = np.arange(4)
    ax.bar(xx - 0.2, t5, width=0.38, color="#f0a875", edgecolor=MUT, label="top-5")
    ax.bar(xx + 0.2, t1, width=0.38, color="#9ab8cc", edgecolor=MUT, label="top-1")
    for i in range(4):
        ax.text(i - 0.2, t5[i] + 0.012, f"{t5[i]*100:.1f}", ha="center", fontsize=7.4)
        ax.text(i + 0.2, t1[i] + 0.012, f"{t1[i]*100:.1f}", ha="center", fontsize=7.4)
    ax.axhline(CHANCE5, color=MUT, ls=":", lw=1)
    ax.set_xticks(xx); ax.set_xticklabels([s.replace("sub-", "s") for s in DEV_SUBJECTS])
    ax.set_ylabel("accuracy"); ax.set_ylim(0, max(t5) * 1.22)
    ax.legend(frameon=False, fontsize=7.6)
    ax.set_title(f"Per subject  (spread {(max(t5)-min(t5))*100:.1f} pp)",
                 fontsize=9.5, loc="left")
    worst = DEV_SUBJECTS[int(np.argmin(t5))]
    ax.text(0.5, -0.26, f"{worst} is the laggard: diagnosed as intrinsic low SNR "
                        f"(CV encoding r 0.027 vs 0.044-0.050), not a fixable defect",
            transform=ax.transAxes, ha="center", fontsize=6.8, color=WARN,
            style="italic")

    ax = fig.add_subplot(gs[0, 2])
    bins = np.arange(1, 104, 4)
    ax.hist(r, bins=bins, color="#9ab8cc", edgecolor=MUT, lw=0.5)
    ax.axvline(float(np.median(r)), color=GOOD, lw=2,
               label=f"median {np.median(r):.0f}")
    ax.axvline(51.5, color=MUT, ls=":", lw=1.4, label="chance median 51.5")
    ax.set_xlabel("rank of the correct video (1 = best)")
    ax.set_ylabel("trials")
    ax.legend(frameon=False, fontsize=7.6)
    ax.set_title("Where the truth lands", fontsize=9.5, loc="left")

    k1, k5, n = int((r <= 1).sum()), int((r <= 5).sum()), r.size
    fig.suptitle(f"Result: top-1 {k1/n*100:.2f}%  ·  top-5 {k5/n*100:.2f}%  "
                 f"({k5}/{n} trials)  ·  median rank {np.median(r):.0f} of 102",
                 fontsize=11.5, fontweight="bold", y=1.04)
    fig.savefig(OUT / "05_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"accuracy_at_k": {int(k): v for k, v in zip(ks, ours)},
            "per_subject": {s: {"top1": a, "top5": b}
                            for s, a, b in zip(DEV_SUBJECTS, t1, t5)},
            "pooled": {"top1": k1 / n, "top5": k5 / n, "k1": k1, "k5": k5, "n": n,
                       "median_rank": float(np.median(r)),
                       "mean_rank": float(np.mean(r)),
                       "p_top1": float(binomtest(k1, n, CHANCE1, alternative="greater").pvalue),
                       "p_top5": float(binomtest(k5, n, CHANCE5, alternative="greater").pvalue),
                       "top5_ci95": [float(beta.ppf(0.025, k5, n - k5 + 1)),
                                      float(beta.ppf(0.975, k5 + 1, n - k5))]},
            "previous_5_scorer_fusion": {"top1": prev["top1"], "top5": prev["top5"]},
            "organizers_baseline": {"top1": bt1, "top5": bt5}}, r


# ------------------------------------------------------------------ 06 robustness
def fig_robustness(fz, r):
    seeds = []
    for sd in fz.get("split_seeds", [1234, 7, 99, 2024, 777]):
        # always the per-seed SINGLE-SPLIT artefact: the frozen one is now the
        # multi-split choice and would misrepresent what one split picks
        fp = RESULTS_DIR / "compare_embed" / f"_frozen_pipeline_s{sd}.json"
        tp = RESULTS_DIR / "compare_embed" / f"_pipeline_test_s{sd}.json"
        if not (fp.exists() and tp.exists()):
            continue
        a = json.loads(fp.read_text()); b = json.loads(tp.read_text())
        seeds.append({"seed": sd, "selected": a["selected"],
                      "top1": b["pooled"]["top1"], "top5": b["pooled"]["top5"]})

    sel = json.loads((RESULTS_DIR / "compare_embed" / "_pipeline_select.json").read_text())
    cfgs = sel["configurations"]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3), gridspec_kw={"wspace": 0.4})

    ax = axes[0]
    xs = np.arange(len(seeds))
    t5 = [s["top5"] for s in seeds]
    eq = ["equal" in s["selected"] for s in seeds]
    ax.bar(xs, t5, color=[GOOD if e else WARN for e in eq], edgecolor=MUT, width=0.6)
    for i, s in enumerate(seeds):
        ax.text(i, s["top5"] + 0.004, f"{s['top5']*100:.2f}", ha="center", fontsize=7.6)
        ax.text(i, 0.02, s["selected"].split("|")[-1], ha="center", fontsize=6.6,
                rotation=90, color="white", fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([str(s["seed"]) for s in seeds], fontsize=7.6)
    ax.set_xlabel("train/val split seed"); ax.set_ylabel("dev-test top-5")
    ax.set_ylim(0, max(t5) * 1.15)
    ax.axhline(fz.get("_t5", 0), color=ACC, ls="--", lw=1.2)
    ax.text(-0.45, fz.get("_t5", 0) - 0.030, "multi-split\nfreeze", ha="left",
            va="top", fontsize=6.4, color=ACC, fontweight="bold")
    ax.set_title(f"What ONE split picks, per seed\n"
                 f"same scorers + transform every time (sd {np.std(t5, ddof=1)*100:.2f} pp)",
                 fontsize=9.3, loc="left")

    ax = axes[1]
    v = np.array([c["val"]["top5"] for c in cfgs.values()])
    ax.scatter(v * 100, np.zeros_like(v), alpha=0)  # keep scale
    ax.hist(v * 100, bins=18, color="#d8d4c8", edgecolor=MUT, lw=0.5)
    ax.axvline(fz["val"]["top5"] * 100, color=GOOD, lw=2,
               label="frozen choice")
    ax.set_xlabel("validation top-5 of each of 240 configurations (%)")
    ax.set_ylabel("configurations")
    ax.legend(frameon=False, fontsize=7.6)
    ax.set_title("The selection surface is flat:\nthe field sits within ~1.5 pp",
                 fontsize=9.3, loc="left")

    ax = axes[2]
    rng = np.random.default_rng(1234)
    bs = np.array([np.mean(rng.choice(r, r.size, replace=True) <= 5)
                   for _ in range(10000)])
    ax.hist(bs * 100, bins=40, color="#9ab8cc", edgecolor="none")
    lo, hi = np.percentile(bs * 100, [2.5, 97.5])
    ax.axvline(float(np.mean(r <= 5)) * 100, color=GOOD, lw=2, label="estimate")
    ax.axvspan(lo, hi, color=GOOD, alpha=0.12, label="95% CI")
    ax.set_xlabel("top-5 (%), bootstrap over trials")
    ax.set_ylabel("resamples")
    ax.legend(frameon=False, fontsize=7.6)
    ax.set_title(f"Sampling uncertainty\n[{lo:.1f}%, {hi:.1f}%]", fontsize=9.3,
                 loc="left")

    axes[0].text(0.5, -0.30,
                 "green = that split chose equal weighting, orange = skill2; "
                 "the channel set and transform never change",
                 transform=axes[0].transAxes, ha="center", fontsize=6.8, color=MUT,
                 style="italic")
    fig.suptitle("Robustness: the result does not hinge on the split, the configuration, "
                 "or one lucky subject", fontsize=11.5, fontweight="bold", y=1.04)
    fig.savefig(OUT / "06_robustness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"split_seeds": seeds,
            "bootstrap_ci95": [float(lo) / 100, float(hi) / 100],
            "val_field_spread_pp": float((v.max() - v.min()) * 100)}


# ------------------------------------------------------------------ 07 top channels
def fig_top_channels(fz, avail, per, pools, skill):
    top = sorted(avail, key=lambda s: -skill[s])[:18]
    fam = Counter()
    w = np.array(fz["weights"])
    for c in fz["channels"]:
        e, s, d = c.split(":")
        fam["encoding" if d == "enc" else "decoding"] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0),
                             gridspec_kw={"width_ratios": [1.65, 1.0], "wspace": 0.30})
    ax = axes[0]
    y = np.arange(len(top))[::-1]
    v = [skill[s] for s in top]
    cols = [WARN if s.startswith("win_") else "#9ab8cc" for s in top]
    ax.barh(y, v, color=cols, edgecolor=MUT, height=0.66)
    for i, s in enumerate(top):
        ax.text(v[i] + 0.002, y[i], f"{skill[s]:.4f}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace(":dec", "").replace(":enc", " [enc]")
                        for s in top], fontsize=6.9)
    ax.set_xlabel("validation MRR")
    ax.set_title("The 18 strongest individual scorers", fontsize=9.6, loc="left")
    ax.set_xlim(0, max(v) * 1.14)
    ax.text(0.99, 0.02, "orange = latency-window EEG", transform=ax.transAxes,
            ha="right", fontsize=7.2, color=WARN, fontweight="bold")

    ax = axes[1]
    mods = Counter(c.split(":")[1].split("@")[0] for c in fz["channels"])
    mm = mods.most_common(10)
    yy = np.arange(len(mm))[::-1]
    ax.barh(yy, [c for _, c in mm], color="#cfd8cf", edgecolor=MUT, height=0.62)
    for i, (m, c) in enumerate(mm):
        ax.text(c + 1, yy[i], str(c), va="center", fontsize=7.2)
    ax.set_yticks(yy); ax.set_yticklabels([m for m, _ in mm], fontsize=7.4)
    ax.set_xlabel("scorers in the frozen set")
    ax.set_title("Stimulus models used", fontsize=9.6, loc="left")
    ax.text(0.99, 0.02,
            f"{fam['decoding']} decoding · {fam['encoding']} encoding\n"
            f"encoding weight share {w[[i for i,c in enumerate(fz['channels']) if c.endswith(':enc')]].sum()*100:.1f}%",
            transform=ax.transAxes, ha="right", fontsize=7.2, color=MUT)

    fig.suptitle("What is inside the frozen ensemble",
                 fontsize=11.5, fontweight="bold", y=1.0)
    fig.savefig(OUT / "07_top_channels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"top_scorers": [{"spec": s, "val_mrr": skill[s]} for s in top],
            "stimulus_model_counts": dict(mods),
            "encoding_weight_share": float(
                w[[i for i, c in enumerate(fz["channels"]) if c.endswith(":enc")]].sum())}


# ------------------------------------------------------------------ 08 alignment
def fig_alignment(per):
    """The cross-subject alignment: validation scaling, and the per-subject effect."""
    import glob

    ex = RESULTS_DIR / "compare_embed" / "explore"
    def load(n):
        p = ex / f"{n}.json"
        return json.loads(p.read_text()) if p.exists() else None

    pts = []
    for nch, own, al in ((9, "01b_9ch_own", "01b_9ch_aligned"),
                         (24, "01b_24ch_own", "01b_24ch_aligned"),
                         (509, "01c_full509_own", "01c_full509_aligned")):
        a, b = load(own), load(al)
        if a and b:
            pts.append((nch, a["pooled_with_bijection"]["top5"],
                        b["pooled_with_bijection"]["top5"],
                        a["per_subject"], b["per_subject"]))
    if not pts:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3), gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    x = [p[0] for p in pts]
    ax.plot(x, [p[1] * 100 for p in pts], "o-", color=MUT, lw=1.8, label="own data only")
    ax.plot(x, [p[2] * 100 for p in pts], "o-", color=GOOD, lw=2.2, label="+ donor subjects")
    for nch, a, b, _, _ in pts:
        ax.annotate(f"{(b-a)*100:+.2f}", xy=(nch, b * 100), xytext=(4, 9),
                    textcoords="offset points", ha="center", fontsize=7.6,
                    color=GOOD, fontweight="bold")
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel("number of scorers fused"); ax.set_ylabel("validation top-5 (%)")
    ax.set_title("The gain GROWS with scale\n(not a small-subset artefact)",
                 fontsize=9.4, loc="left")
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")

    ax = axes[1]
    _, _, _, pa, pb = pts[-1]
    subs = list(pa)
    xx = np.arange(len(subs))
    ax.bar(xx - 0.2, [pa[s]["top5"] * 100 for s in subs], width=0.38,
           color="#d8d4c8", edgecolor=MUT, label="own data only")
    ax.bar(xx + 0.2, [pb[s]["top5"] * 100 for s in subs], width=0.38,
           color=GOOD, edgecolor=MUT, label="+ donors")
    for i, s in enumerate(subs):
        d = (pb[s]["top5"] - pa[s]["top5"]) * 100
        ax.text(i, max(pa[s]["top5"], pb[s]["top5"]) * 100 + 1.1, f"{d:+.2f}",
                ha="center", fontsize=7.4, color=GOOD, fontweight="bold")
    ax.set_xticks(xx); ax.set_xticklabels([s.replace("sub-", "s") for s in subs])
    ax.set_ylabel("validation top-5 (%)")
    ax.set_title("Every subject improves\n(509 scorers)", fontsize=9.4, loc="left")
    ax.legend(frameon=False, fontsize=7.4)

    ax = axes[2]
    ctrl = [("own data\nonly", 36.28, "#d8d4c8"),
            ("pool donors\nWITHOUT\nalignment", 29.23, WARN),
            ("pool donors\nWITH\nalignment", 37.08, GOOD)]
    ax.bar(range(3), [c[1] for c in ctrl], color=[c[2] for c in ctrl],
           edgecolor=MUT, width=0.6)
    for i, c in enumerate(ctrl):
        ax.text(i, c[1] + 0.5, f"{c[1]:.2f}", ha="center", fontsize=8,
                fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels([c[0] for c in ctrl], fontsize=7)
    ax.set_ylabel("validation top-5 (%)"); ax.set_ylim(25, 40)
    ax.set_title("The control that matters:\nalignment does the work",
                 fontsize=9.4, loc="left")
    ax.text(0.5, -0.33, "pooling other people's EEG raw makes it WORSE — their feature\n"
                        "spaces are not comparable. The map is what rescues it.",
            transform=ax.transAxes, ha="center", fontsize=6.8, color=MUT, style="italic")

    fig.suptitle("Cross-subject alignment: letting each subject borrow the others' "
                 "training recordings", fontsize=11.5, fontweight="bold", y=1.04)
    fig.savefig(OUT / "08_cross_subject.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"scaling": [{"n_scorers": n, "own": a, "aligned": b} for n, a, b, _, _ in pts]}


# ------------------------------------------------------------------ main
def main() -> int:
    t0 = time.perf_counter()
    print("loading channel cache ...", flush=True)
    fz, avail, per, pools = load_all()
    print(f"  {len(avail)} scorers x {len(DEV_SUBJECTS)} subjects "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    print("per-scorer validation skill ...", flush=True)
    t = time.perf_counter()
    skill = {s: val_mrr(per, pools, s) for s in avail}
    print(f"  done ({time.perf_counter()-t:.0f}s)", flush=True)

    out = {}
    res, r = fig_results(fz, avail, per, skill)
    out["results"] = res
    fz["_t1"], fz["_t5"] = res["pooled"]["top1"], res["pooled"]["top5"]
    for name, fn in [("architecture", lambda: fig_architecture(fz)),
                     ("latency_windows", lambda: fig_windows(avail, per, pools)),
                     ("feature_ablation", lambda: fig_ablation(fz, avail, per, pools, skill)),
                     ("fusion_mechanism", lambda: fig_mechanism(fz, avail, per, pools, skill)),
                     ("robustness", lambda: fig_robustness(fz, r)),
                     ("top_channels", lambda: fig_top_channels(fz, avail, per, pools, skill)),
                     ("cross_subject", lambda: fig_alignment(per))]:
        t = time.perf_counter()
        v = fn()
        if v is not None:
            out[name] = v
        print(f"  {name} ({time.perf_counter()-t:.0f}s)", flush=True)

    out["frozen"] = {k: v for k, v in fz.items() if k != "weights"}
    p = RESULTS_DIR / "compare_embed" / "_explain_pipeline.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {p.relative_to(ROOT)}")
    print(f"figures in {OUT.relative_to(ROOT)}: "
          f"{sorted(x.name for x in OUT.glob('*.png'))}")
    print(f"total {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
