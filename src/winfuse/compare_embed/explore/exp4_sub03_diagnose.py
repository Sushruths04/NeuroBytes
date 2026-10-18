"""PART A - diagnose what is different about sub-03. TRAINING DATA ONLY.

Nothing here reads tst/ or tst_stim/. Every quantity is computed from
pp.preprocessed_train(sub), i.e. the subject's own training epochs
(n, 17, 750) @ 250 Hz, t=0 = video onset, no pre-stimulus baseline.

Measures, all per subject, all comparable across subjects:
  1. per-channel std and its uniformity (max/min ratio, CV across channels)
  2. per-trial and per-channel outlier/artefact rate under a k*MAD criterion
  3. channel-channel correlation structure: dead/duplicated channels,
     mean |corr|, condition number of the covariance
  4. split-half reliability of the evoked response (odd vs even trials,
     Spearman-Brown corrected), overall and per channel
  5. evoked-response SNR as a function of time after onset
  6. power spectra per channel (line noise, drift, muscle)
  7. whether a handful of extreme trials dominate the total variance

Writes results/compare_embed/explore/exp4_sub03_diagnose.json and prints a
side-by-side table.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.signal import welch

from winfuse.common import preprocess as pp
from winfuse.common.config import RESULTS_DIR, SFREQ

SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04"]
OUT = RESULTS_DIR / "compare_embed" / "explore"
OUT.mkdir(parents=True, exist_ok=True)


def _spearman_brown(r: float) -> float:
    """Correct a half-length reliability to full length."""
    if r >= 1.0:
        return 1.0
    return (2.0 * r) / (1.0 + r) if (1.0 + r) != 0 else 0.0


def diagnose(sub: str) -> dict:
    X, ids = pp.preprocessed_train(sub)          # (n, 17, 750) float32, z-scored
    X = np.asarray(X, dtype=np.float64)
    n, nch, nt = X.shape

    # ---- 1. per-channel std (over all trials x time) and uniformity
    ch_std = X.std(axis=(0, 2))                  # (17,)
    std_ratio = float(ch_std.max() / max(ch_std.min(), 1e-12))
    std_cv = float(ch_std.std() / max(ch_std.mean(), 1e-12))

    # per-channel std measured per-trial, so a dead channel on a few trials shows up
    per_trial_ch_std = X.std(axis=2)             # (n, 17)
    # a "dead" channel = near-zero variance on a trial
    dead_frac = float(np.mean(per_trial_ch_std < 1e-3))

    # ---- 2. outlier / artefact rate under k*MAD, per channel then pooled
    # criterion applied to per-trial peak absolute amplitude per channel
    peak = np.abs(X).max(axis=2)                 # (n, 17) peak |amp| per trial-channel
    med = np.median(peak, axis=0)                # (17,)
    mad = np.median(np.abs(peak - med), axis=0) * 1.4826 + 1e-12
    k = 5.0
    outlier_mask = peak > (med + k * mad)        # (n, 17)
    outlier_rate_per_ch = outlier_mask.mean(axis=0)        # (17,)
    outlier_rate = float(outlier_mask.mean())
    # fraction of TRIALS with at least one outlier channel
    trial_outlier_rate = float(np.mean(outlier_mask.any(axis=1)))

    # ---- 3. channel-channel correlation structure
    # flatten (trial,time) -> per-channel long vector, correlate channels
    Xc = X.transpose(1, 0, 2).reshape(nch, -1)   # (17, n*750)
    C = np.corrcoef(Xc)                          # (17,17)
    off = C[~np.eye(nch, dtype=bool)]
    mean_abs_corr = float(np.mean(np.abs(off)))
    max_abs_corr = float(np.max(np.abs(off)))    # near 1.0 => a duplicated channel
    # covariance condition number: large => near-collinear / rank-deficient channel set
    cov = np.cov(Xc)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 1e-12, None)
    cond = float(ev.max() / ev.min())
    # effective rank (participation ratio of eigenvalues)
    eff_rank = float((ev.sum() ** 2) / (ev ** 2).sum())

    # ---- 4. split-half reliability of the evoked response (odd/even trials)
    # evoked = mean over trials; correlate odd-trial evoked vs even-trial evoked
    odd = X[1::2]
    even = X[0::2]
    ev_odd = odd.mean(axis=0)                    # (17, 750)
    ev_even = even.mean(axis=0)                  # (17, 750)
    # overall: flatten channels x time
    r_overall = float(np.corrcoef(ev_odd.ravel(), ev_even.ravel())[0, 1])
    rel_overall = _spearman_brown(r_overall)
    # per channel
    rel_per_ch = []
    for c in range(nch):
        rc = np.corrcoef(ev_odd[c], ev_even[c])[0, 1]
        rel_per_ch.append(_spearman_brown(float(rc)))
    rel_per_ch = np.array(rel_per_ch)

    # ---- 5. evoked SNR as a function of time
    # signal = evoked (mean over trials); noise = sd across trials / sqrt(n)
    evoked = X.mean(axis=0)                       # (17, 750)
    noise = X.std(axis=0) / np.sqrt(n)            # (17, 750) SEM of the mean
    snr_t = (np.abs(evoked) / np.maximum(noise, 1e-12)).mean(axis=0)   # (750,)
    # summarise: peak SNR, time of peak (ms), mean SNR in 50-400 ms (the pipeline window)
    t_ms = np.arange(nt) / SFREQ * 1000.0
    peak_snr = float(snr_t.max())
    peak_snr_ms = float(t_ms[int(np.argmax(snr_t))])
    win = (t_ms >= 50) & (t_ms < 400)
    mean_snr_50_400 = float(snr_t[win].mean())
    # global field power of the evoked response (how large the evoked response is)
    gfp = float(np.sqrt((evoked ** 2).mean()))
    # evoked energy as a function of time (RMS over channels)
    gfp_t = np.sqrt((evoked ** 2).mean(axis=0))   # (750,)
    gfp_peak_ms = float(t_ms[int(np.argmax(gfp_t))])

    # ---- 6. power spectra (line noise / drift / muscle)
    # Welch per channel then average; report band fractions
    f, Pxx = welch(X.reshape(n * nch, nt), fs=SFREQ, nperseg=min(256, nt), axis=-1)
    Pxx = Pxx.reshape(n, nch, -1).mean(axis=(0, 1))     # (nfreq,)
    tot = Pxx.sum()

    def band_frac(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(Pxx[m].sum() / max(tot, 1e-12))

    drift_frac = band_frac(0.0, 2.0)             # excessive => drift
    low_frac = band_frac(0.5, 8.0)
    alpha_frac = band_frac(8.0, 13.0)
    muscle_frac = band_frac(20.0, 40.0)          # high => muscle/EMG
    # line noise: energy near 50 Hz is band-passed out at 40, but check residual slope

    # ---- 7. do a few extreme trials dominate the variance?
    trial_energy = (X ** 2).mean(axis=(1, 2))    # (n,) mean power per trial
    order = np.sort(trial_energy)[::-1]
    frac_top1pct = float(order[:max(1, n // 100)].sum() / order.sum())
    frac_top5pct = float(order[:max(1, n // 20)].sum() / order.sum())
    # gini-like: ratio of max trial energy to median
    energy_med = float(np.median(trial_energy))
    energy_max_over_med = float(trial_energy.max() / max(energy_med, 1e-12))

    return {
        "subject": sub,
        "n_trials": int(n),
        "ch_std": ch_std.round(5).tolist(),
        "ch_std_min": float(ch_std.min()),
        "ch_std_max": float(ch_std.max()),
        "ch_std_ratio": std_ratio,
        "ch_std_cv": std_cv,
        "dead_channel_trial_frac": dead_frac,
        "outlier_rate_5mad": outlier_rate,
        "trial_outlier_rate_5mad": trial_outlier_rate,
        "outlier_rate_per_ch": outlier_rate_per_ch.round(5).tolist(),
        "mean_abs_chan_corr": mean_abs_corr,
        "max_abs_chan_corr": max_abs_corr,
        "cov_condition_number": cond,
        "cov_effective_rank": eff_rank,
        "reliability_overall_sb": rel_overall,
        "reliability_overall_raw_r": r_overall,
        "reliability_per_ch_sb": rel_per_ch.round(4).tolist(),
        "reliability_per_ch_mean": float(rel_per_ch.mean()),
        "reliability_per_ch_min": float(rel_per_ch.min()),
        "reliability_per_ch_max": float(rel_per_ch.max()),
        "evoked_peak_snr": peak_snr,
        "evoked_peak_snr_ms": peak_snr_ms,
        "evoked_mean_snr_50_400ms": mean_snr_50_400,
        "evoked_gfp": gfp,
        "evoked_gfp_peak_ms": gfp_peak_ms,
        "spec_drift_frac_0_2hz": drift_frac,
        "spec_low_frac_0.5_8hz": low_frac,
        "spec_alpha_frac_8_13hz": alpha_frac,
        "spec_muscle_frac_20_40hz": muscle_frac,
        "energy_frac_top1pct_trials": frac_top1pct,
        "energy_frac_top5pct_trials": frac_top5pct,
        "energy_max_over_median": energy_max_over_med,
    }


def main() -> int:
    res = {s: diagnose(s) for s in SUBJECTS}
    (OUT / "exp4_sub03_diagnose.json").write_text(json.dumps(res, indent=2))

    def row(label, key, fmt="{:.4f}", scale=1.0):
        vals = []
        for s in SUBJECTS:
            v = res[s][key]
            vals.append(fmt.format(v * scale))
        print(f"  {label:34} " + "".join(f"{v:>12}" for v in vals))

    print("\n" + "=" * 86)
    print("  PART A  sub-03 diagnosis (TRAINING DATA ONLY)")
    print("=" * 86)
    print(f"  {'measure':34} " + "".join(f"{s:>12}" for s in SUBJECTS))
    print("  " + "-" * 82)
    row("n trials", "n_trials", "{:.0f}")
    print("  -- channel variance / uniformity --")
    row("per-ch std min", "ch_std_min")
    row("per-ch std max", "ch_std_max")
    row("per-ch std max/min ratio", "ch_std_ratio")
    row("per-ch std CV across channels", "ch_std_cv")
    row("dead-channel trial frac", "dead_channel_trial_frac")
    print("  -- outliers (5*MAD on per-trial peak) --")
    row("outlier rate (ch x trial)", "outlier_rate_5mad", "{:.4f}")
    row("trials with any outlier ch", "trial_outlier_rate_5mad", "{:.4f}")
    print("  -- channel correlation structure --")
    row("mean |chan corr|", "mean_abs_chan_corr")
    row("max |chan corr|", "max_abs_chan_corr")
    row("cov condition number", "cov_condition_number", "{:.1f}")
    row("cov effective rank (of 17)", "cov_effective_rank", "{:.2f}")
    print("  -- evoked-response reliability (odd/even, Spearman-Brown) --")
    row("reliability overall", "reliability_overall_sb")
    row("reliability per-ch mean", "reliability_per_ch_mean")
    row("reliability per-ch min", "reliability_per_ch_min")
    row("reliability per-ch max", "reliability_per_ch_max")
    print("  -- evoked SNR vs time --")
    row("peak SNR", "evoked_peak_snr", "{:.3f}")
    row("peak SNR time (ms)", "evoked_peak_snr_ms", "{:.0f}")
    row("mean SNR 50-400 ms", "evoked_mean_snr_50_400ms", "{:.3f}")
    row("evoked GFP (RMS)", "evoked_gfp", "{:.4f}")
    row("evoked GFP peak time (ms)", "evoked_gfp_peak_ms", "{:.0f}")
    print("  -- power spectrum --")
    row("drift frac (0-2 Hz)", "spec_drift_frac_0_2hz")
    row("low frac (0.5-8 Hz)", "spec_low_frac_0.5_8hz")
    row("alpha frac (8-13 Hz)", "spec_alpha_frac_8_13hz")
    row("muscle frac (20-40 Hz)", "spec_muscle_frac_20_40hz")
    print("  -- extreme-trial dominance --")
    row("energy frac top 1% trials", "energy_frac_top1pct_trials")
    row("energy frac top 5% trials", "energy_frac_top5pct_trials")
    row("max/median trial energy", "energy_max_over_median", "{:.2f}")
    print("=" * 86)

    # Per-channel reliability detail for sub-03 vs the mean of the others
    print("\n  per-channel split-half reliability (Spearman-Brown):")
    print(f"  {'ch':>3} " + "".join(f"{s:>10}" for s in SUBJECTS))
    for c in range(17):
        vals = [res[s]["reliability_per_ch_sb"][c] for s in SUBJECTS]
        flag = "  <-- low" if vals[2] < 0.2 and vals[2] < min(vals[0], vals[1], vals[3]) else ""
        print(f"  {c:>3} " + "".join(f"{v:>10.4f}" for v in vals) + flag)

    print(f"\n  wrote {(OUT / 'exp4_sub03_diagnose.json').relative_to(RESULTS_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
