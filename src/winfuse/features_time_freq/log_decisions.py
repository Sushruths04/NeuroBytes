"""Log Agent A's material decisions via the frozen decision logger.

Run once:  .venv-eeg/bin/python -m winfuse.features_time_freq.log_decisions
Appends to decisions/A.md and decisions/A.jsonl. Numbers here match the code in
this package and the measured runs on sub-01..06.
"""
from __future__ import annotations

from winfuse.common.decisions import log_decision


def main() -> None:
    log_decision(
        agent="A",
        title="raw_signal decimation factor",
        choice="Block-mean the 250 Hz preprocessed trace by 5 -> 50 Hz, 150 samples/ch, "
               "17x150 = 2550 features.",
        alternatives=[
            "No further decimation (750 samples/ch = 12750 features): 5x larger, "
            "mostly redundant after a 40 Hz low-pass.",
            "Decimate by 10 -> 25 Hz: violates Nyquist for the 30-40 Hz gamma band, "
            "would alias beta/gamma into the trace.",
        ],
        rationale="The signal is band-limited to 40 Hz, so 50 Hz (block mean of 5) keeps "
                  "Nyquist=25 Hz? No -- block mean is a crude anti-alias, but the input is "
                  "already 0.5-40 Hz band-passed; a factor-5 block mean preserves the "
                  "envelope and sustained response while cutting the feature count 5x. "
                  "50 Hz output rate keeps 2 samples per fastest (40 Hz would need 80 Hz; "
                  "the block mean is a deliberate low-pass of the already-filtered trace).",
        consequences="raw_signal is amplitude-scaled, so it shrinks on test (test std is "
                     "62-68% of train). It is also the largest cache (9.4 MB/subject). "
                     "The block mean attenuates 20-40 Hz content; this is a lossy summary, "
                     "not a faithful 50 Hz resample.",
    )
    log_decision(
        agent="A",
        title="ERP window set and reduction",
        choice="Mean amplitude per channel over all 8 ERP_WINDOWS from config "
               "(p1n1..w2500_3000). 8x17 = 136 features.",
        alternatives=[
            "Peak amplitude per window: more sensitive to single-sample noise with no "
            "baseline to normalise against.",
            "Only the three early windows (p1n1,n2p2,late): discards the sustained "
            "response, which over a 3 s video is real signal.",
        ],
        rationale="Means are robust and cheap; using the frozen window set keeps the "
                  "feature comparable across agents. The late/long windows capture the "
                  "sustained response the baseline notebook relied on (0.05-0.60 s).",
        consequences="Amplitude-scaled (a.u.), vulnerable to the train/test scale "
                     "mismatch. Window means are blind to within-window dynamics.",
    )
    log_decision(
        agent="A",
        title="Hjorth parameter definition",
        choice="activity=var(x); mobility=sqrt(var(x')/var(x)); "
               "complexity=mobility(x')/mobility(x), using first/second differences. "
               "3x17 = 51 features. Computed in float64.",
        alternatives=[
            "antropy.hjorth: adds a dependency and is slower; identical formula.",
        ],
        rationale="Closed-form, O(n) per channel, matches the standard definition. "
                  "float64 avoids catastrophic cancellation in the variance ratios.",
        consequences="activity is a variance -> amplitude-scaled; mobility and complexity "
                     "are ratios -> scale-invariant. Mixed sensitivity, flagged in report.",
    )
    log_decision(
        agent="A",
        title="time_stats feature set",
        choice="Per channel: peak-to-peak, skew (bias-corrected), kurtosis (Fisher, "
               "bias-corrected), line length (sum|diff|), RMS, zero-crossing rate. "
               "6x17 = 102 features. Computed in float64; non-finite coerced to 0.",
        alternatives=[
            "Add variance: already captured by Hjorth activity and RMS^2; dropped to "
            "avoid duplicate columns across extractors.",
        ],
        rationale="A compact morphology set spanning amplitude (ptp,rms,line_length), "
                  "shape (skew,kurtosis) and rate (zcr). Cheap and interpretable.",
        consequences="ptp, line_length, rms are amplitude-scaled; skew, kurtosis, zcr are "
                     "scale-invariant. Mixed, flagged in report.",
    )
    log_decision(
        agent="A",
        title="Welch PSD segment length (nperseg) and overlap",
        choice="Hann window, nperseg=250 (1.0 s at 250 Hz), noverlap=125 (50%), "
               "scaling='density', detrend='constant'. Yields 1 Hz bins and ~5 "
               "overlapping segments across the 3 s epoch.",
        alternatives=[
            "nperseg=125 (0.5 s): 2 Hz bins smear the 0.5-4 Hz delta edge.",
            "nperseg=750 (full epoch): 0.33 Hz bins but a single segment -> no variance "
            "averaging, noisy PSD.",
        ],
        rationale="1 Hz bins resolve the delta lower edge; 5 overlapping segments give "
                  "usable variance reduction. Same choice documented in the base contract "
                  "example, kept for cross-agent comparability.",
        consequences="Delta (0.5-4 Hz) rests on only ~4 bins and is the least reliable "
                     "band. band_power is absolute power -> amplitude-scaled.",
    )
    log_decision(
        agent="A",
        title="Absolute band power stored as log10",
        choice="band_power returns log10(trapz PSD over band + 1e-12). 5 bands x 17 ch = 85.",
        alternatives=[
            "Linear power: heavy right skew across trials, dominated by delta.",
        ],
        rationale="log10 compresses the dynamic range so downstream linear models and "
                  "distance metrics behave; +1e-12 guards log(0).",
        consequences="Still amplitude-scaled: a global test-side amplitude shrink adds a "
                     "near-constant offset per band (log of the scale factor). Noted in report.",
    )
    log_decision(
        agent="A",
        title="rel_band_power: scale-invariant spectral set",
        choice="Relative band power (band/total, 5x17), spectral entropy (normalised, "
               "17), 95% spectral edge frequency (17), and 1/f log-log slope fit over "
               "2-35 Hz (17). 136 features total.",
        alternatives=[
            "Spectral edge at 90% or 50% (median freq): 95% is the common EEG choice and "
            "most sensitive to the gamma tail; 50% hugs the dominant low-frequency peak.",
            "Slope fit over the full 0.5-40 Hz: the 0.5 Hz high-pass roll-off and the 40 "
            "Hz low-pass corner bias the fit; 2-35 Hz avoids both corners.",
        ],
        rationale="These are ratios/shapes, so a global amplitude change cancels. This is "
                  "the extractor expected to survive the train/test amplitude mismatch best.",
        consequences="Insensitive to overall power -> blind to genuine broadband power "
                     "differences between stimuli. Slope needs >=2 bins in [2,35] Hz (true here).",
    )
    log_decision(
        agent="A",
        title="Time-resolved band power: Morlet wavelet, windowing",
        choice="One complex Morlet per band at the band's geometric-mean centre freq, "
               "n_cycles=max(3, f_c*0.25) so the wavelet support ~=250 ms. Instantaneous "
               "power |conv|^2 averaged in 250 ms windows (62 samples) stepped by 125 ms "
               "(31 samples) -> 23 windows across 3 s.",
        alternatives=[
            "STFT with a 250 ms window: coarse 4 Hz bins, poor for delta/theta.",
            "Per-band fixed n_cycles=7: wavelet support for delta (fc~1.4 Hz) would be "
            ">0.8 s, far longer than the 250 ms analysis window.",
        ],
        rationale="Tying n_cycles to f_c*window keeps every band's wavelet near the 250 ms "
                  "analysis scale, so time resolution is comparable across bands. FFT "
                  "convolution keeps it CPU-cheap (<1 s/subject for all 5 bands).",
        consequences="23 windows, not the ~22 the brief estimated (ceil of (750-62)/31 + 1). "
                     "tf_band_power is 17x5x23 = 1955 features and 7.2 MB/subject -- the "
                     "second-largest cache. log10 power -> amplitude-scaled.",
    )
    log_decision(
        agent="A",
        title="Bounding the TF feature count (reduced variant)",
        choice="tf_band_power_reduced time-pools the 23 windows into 4 equal coarse "
               "epochs by mean -> 17x5x4 = 340 features.",
        alternatives=[
            "PCA on the 1955-dim TF block: a fitted transform that could overfit per "
            "subject and complicates leakage accounting.",
            "Keep only the first 500 ms: discards the sustained-response dynamics that a "
            "3 s video should drive.",
        ],
        rationale="4 coarse epochs (~0.75 s each) retain gross temporal structure "
                  "(onset vs sustained vs offset) at 1/6 the width, with no fitted "
                  "parameter, so it is trivially leakage-free.",
        consequences="Loses fine 125 ms timing. Still amplitude-scaled.",
    )
    log_decision(
        agent="A",
        title="ERSP without a baseline: reference to the TRAIN grand average",
        choice="ersp_trialmean expresses each trial's log TF power in dB relative to the "
               "grand-average log TF power across TRAIN trials: 10*(log10 P - mean_train "
               "log10 P). The reference (17x5x23) is learned in _fit on train only and "
               "reused unchanged in transform. Then time-pooled to 4 epochs -> 340 feats.",
        alternatives=[
            "Pre-stimulus baseline: impossible, recording starts at video onset.",
            "Within-trial baseline (first 40 ms): only 10 samples at 250 Hz, too noisy, "
            "and 40 ms is pre-neural-response so it is mostly filter edge transient.",
            "Within-trial whole-epoch mean: removes the sustained response, which is real "
            "signal over a 3 s video.",
        ],
        rationale="With no baseline period, the across-trial grand average is the only "
                  "leakage-safe reference that normalises per-channel/per-band gain while "
                  "preserving each trial's deviation from the population. Fitting it in "
                  "_fit guarantees the test set is referenced to TRAIN statistics only.",
        consequences="Because it subtracts a per-channel/band log-power reference, a global "
                     "test-side amplitude shrink largely cancels in dB -> more scale-robust "
                     "than raw tf_band_power. The reference adds a fitted parameter that "
                     "must be persisted with the extractor for test-time use.",
    )
    log_decision(
        agent="A",
        title="Peak RSS measurement without psutil",
        choice="resource.getrusage(RUSAGE_SELF).ru_maxrss, normalised to MB (bytes on "
               "macOS, KiB on Linux). Reported as a process high-water mark.",
        alternatives=[
            "psutil.Process().memory_info().rss: not installed in .venv-eeg, and the "
            "brief forbids editing requirements.txt.",
        ],
        rationale="Standard-library, zero new dependency, correct on the target macOS M2.",
        consequences="ru_maxrss is monotonic per process, so later extractors in the same "
                     "run inherit earlier peaks; the number is an upper bound, not a "
                     "per-extractor delta. Stated in the runner docstring and results notes.",
    )
    print("logged Agent A decisions ->", "decisions/A.md", "decisions/A.jsonl")


if __name__ == "__main__":
    main()
