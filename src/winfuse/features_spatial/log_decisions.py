"""One-shot logger for Agent B's material decisions. Idempotent-ish: re-running
appends duplicates, so run once. Invoked as:

    .venv-eeg/bin/python -m winfuse.features_spatial.log_decisions
"""
from __future__ import annotations

from winfuse.common.decisions import log_decision

A = "B"


def main() -> None:
    log_decision(
        agent=A,
        title="Covariance estimator and conditioning",
        choice="pyriemann Covariances(estimator='oas') — Oracle Approximating "
               "Shrinkage — for every covariance-based extractor.",
        alternatives=[
            "Empirical ('scm'): unbiased but can be near-singular on low-variance "
            "channels, which breaks the Riemannian log-map.",
            "Hand-tuned Ledoit-Wolf / fixed ridge: another hyperparameter to justify.",
        ],
        rationale="17 channels x 750 samples is ~44 samples/channel, so the sample "
                  "covariance is already well-conditioned. OAS applies a small, "
                  "analytically-chosen shrinkage that guarantees SPD matrices for the "
                  "Riemannian metric with no manual regulariser.",
        consequences="Features are slightly biased toward the identity vs the raw "
                     "empirical covariance; the shrinkage is data-driven and identical "
                     "across subjects for comparability.",
        tags=["covariance", "spatial"],
    )
    log_decision(
        agent=A,
        title="Tangent-space reference mean",
        choice="Riemannian geometric mean of the subject's TRAIN covariances, fitted "
               "once in cov_tangent._fit and reused for transform.",
        alternatives=[
            "Identity reference (no fitting): ignores the subject's covariance "
            "distribution, inflating tangent-space distortion.",
            "Per-split means: would leak across the eventual train/test boundary.",
        ],
        rationale="Projecting at the geometric mean minimises linearisation error of "
                  "the log-map over the data actually seen, and is fitted on train only.",
        consequences="The reference differs per subject, so raw cov_tangent features "
                     "are NOT directly comparable across subjects — that is what "
                     "cov_tangent_ea addresses.",
        tags=["riemann", "tangent-space"],
    )
    log_decision(
        agent=A,
        title="Euclidean alignment across subjects without paired trials",
        choice="cov_tangent_ea whitens each trial covariance by Cref^{-1/2}, where "
               "Cref is that subject's own TRAIN Riemannian mean, then projects to the "
               "tangent space at the identity.",
        alternatives=[
            "Align paired trials across subjects: IMPOSSIBLE — no two subjects saw the "
            "same stimulus set and there are no within-subject repeats.",
            "A single shared reference across subjects: would require pooling raw data "
            "and still not fix per-subject scale.",
        ],
        rationale="Whitening by the per-subject reference maps every subject's mean "
                  "covariance to ~identity, so the tangent features share a common "
                  "origin and scale. Alignment is on covariance STATISTICS, not paired "
                  "trials, which is the only option given disjoint stimulus sets.",
        consequences="Removes the subject-specific covariance offset; genuine "
                     "subject-level differences in overall scale are also removed by "
                     "design.",
        tags=["euclidean-alignment", "cross-subject"],
    )
    log_decision(
        agent=A,
        title="No class labels for xDAWN/CSP — unsupervised evoked-subspace analogue",
        choice="xdawn_evoked solves the generalised eigenproblem "
               "Cov(grand-average ERP) w = lambda Cov(total) w on TRAIN, keeping the "
               "top-6 evoked-SNR filters; features = log-variance of the filtered "
               "component time courses.",
        alternatives=[
            "Supervised xDAWN/CSP with invented pseudo-labels: REJECTED — every trial "
            "is a different stimulus, so any label is fabricated.",
            "Binary evoked-vs-induced contrast as a pseudo-target: still needs an "
            "arbitrary threshold; the across-trial average captures the same "
            "phase-locked subspace without one.",
        ],
        rationale="xDAWN's signal model estimates the evoked response as the "
                  "across-trial average, which needs only the common t=0 onset, not "
                  "class labels. This is the honest unsupervised analogue.",
        consequences="Captures only phase-locked (evoked) structure; induced "
                     "oscillatory responses are treated as noise and down-weighted.",
        tags=["xdawn", "no-labels"],
    )
    log_decision(
        agent=A,
        title="PCA / ICA component count",
        choice="spatial_pca keeps the smallest number of components explaining >=99% "
               "of channel-space variance (<=17). ica_components (FastICA) uses the "
               "same 99% rank, capped at 15.",
        alternatives=[
            "Fixed 17 components: keeps near-zero-variance directions that add noise.",
            "Scree/elbow by eye: not reproducible.",
        ],
        rationale="A variance threshold is reproducible and data-driven; 99% keeps the "
                  "spatial structure while dropping the numerically degenerate tail. "
                  "FastICA requires n_components <= whitening rank.",
        consequences="Component count can differ slightly per subject; it is recorded "
                     "in params.n_components for each run.",
        tags=["pca", "ica"],
    )
    log_decision(
        agent=A,
        title="ICA algorithm, seed and max_iter",
        choice="sklearn FastICA (parallel, logcosh), whiten='unit-variance', "
               "random_state=SEED (1234), max_iter=500.",
        alternatives=[
            "MNE Infomax/Picard: picard is not installed; Infomax is slower with no "
            "accuracy gain for 17 channels.",
            "Default max_iter=200: FastICA occasionally fails to converge on EEG "
            "covariance structure.",
        ],
        rationale="FastICA is deterministic under a fixed seed and fast for 17 "
                  "channels; max_iter=500 gives convergence headroom.",
        consequences="A non-converged fit would still return components; convergence "
                     "is not asserted here, only bounded by max_iter.",
        tags=["ica", "reproducibility"],
    )
    log_decision(
        agent=A,
        title="Nonlinear entropy parameters and sample-entropy subsampling",
        choice="permutation entropy order=3 (normalised); sample entropy order=2 on "
               "the first 375 samples (1.5 s at 250 Hz); Higuchi/Petrosian FD and DFA "
               "on the full 750 samples. Measured cost on sub-01 (968 trials x 17 "
               "channels): transform = 4.95 s total (~5.1 ms/trial).",
        alternatives=[
            "Sample entropy on the full 750 samples: O(n^2) embedding search; the "
            "window halves that cost and keeps the metric comparable across trials.",
            "Larger permutation order (4-5): 24-120 ordinal patterns need far more "
            "than 750 samples to populate reliably.",
        ],
        rationale="antropy's sample_entropy is Numba-accelerated so even full-length "
                  "is only ~0.5 ms/channel, but bounding it to 1.5 s caps worst-case "
                  "runtime and fixes the window length so trials are comparable. "
                  "order=3 permutation entropy (6 patterns) is well-populated by 750 "
                  "samples.",
        consequences="Sample entropy reflects only the first half of the epoch; the "
                     "window length is recorded in params.sampen_window_samples. "
                     "Near-constant channels yield inf sample entropy / failed DFA; "
                     "these are imputed with the per-feature median so the finiteness "
                     "contract holds (noted in reports/B_spatial_viz.md).",
        tags=["entropy", "antropy", "timing"],
    )
    log_decision(
        agent=A,
        title="Peak RSS measurement without psutil",
        choice="resource.getrusage(RUSAGE_SELF).ru_maxrss, interpreted as bytes on "
               "Darwin and kilobytes on Linux.",
        alternatives=[
            "psutil.Process().memory_info().rss: psutil is not installed in "
            ".venv-eeg.",
            "tracemalloc: measures Python allocations only, misses numpy/BLAS buffers.",
        ],
        rationale="ru_maxrss is a stdlib peak-RSS figure that includes native "
                  "allocations; the Darwin/Linux unit difference is handled explicitly.",
        consequences="ru_maxrss is a high-water mark for the whole process, so values "
                     "are monotonic across methods within one run rather than "
                     "per-method deltas (noted in the report).",
        tags=["profiling", "memory"],
    )
    log_decision(
        agent=A,
        title="Figures use channel-index heat maps, never scalp topomaps",
        choice="All spatial figures are channel x channel or channel x time heat maps "
               "and index-labelled traces.",
        alternatives=[
            "mne.viz.plot_topomap: IMPOSSIBLE — the data has no channel names and no "
            "montage, so there is no 2-D sensor geometry to interpolate onto.",
        ],
        rationale="Channels are anonymous indices 0-16 (reports/00_data_inspection.md "
                  "§7); any topomap would require inventing sensor positions.",
        consequences="Spatial structure is shown as covariance/heat matrices only; "
                     "scalp-space interpretation is unavailable.",
        tags=["viz", "montage"],
    )
    log_decision(
        agent=A,
        title="compare.py handles missing Agent-C metrics gracefully",
        choice="compare.py reads results/*/metrics.json via load_all and, for each "
               "chart (blink / split-half / retrieval), plots only methods that have "
               "that key; methods lacking it are listed as skipped and omitted.",
        alternatives=[
            "Fail if any method is missing a metric: breaks while Agent C is still "
            "running.",
            "Fabricate zeros for missing metrics: violates the no-fabrication rule.",
        ],
        rationale="Agents run concurrently; validation metrics may not exist yet. "
                  "Omitting absent keys keeps the figure honest and non-crashing.",
        consequences="Early charts may show few/no bars; a synthetic fixture proves "
                     "the full charts render once metrics arrive.",
        tags=["viz", "robustness"],
    )
    print("Agent B decisions logged to decisions/B.md")


if __name__ == "__main__":
    main()
