"""Log Agent V-C's material decisions. Idempotent (safe to re-run).

    .venv-embed/bin/python -m winfuse.compare_embed.log_decisions
"""
from __future__ import annotations

from winfuse.common.decisions import log_decision

A = "V-C"
T = ["phase2", "compare_embed"]


def main() -> None:
    log_decision(
        agent=A, title="EEG feature sets included (and the two dropped)",
        choice="7 Phase-1 extractors: raw_signal, erp_windows, band_power, "
               "rel_band_power, tf_band_power_reduced, xdawn_evoked, cov_tangent; "
               "plus the preprocessed (n,17,750) array for the latency analysis.",
        alternatives=[
            "Include cov_tangent_ea and ersp_trialmean: rejected — results/_redundancy "
            "proved them per-column-affine-identical to cov_tangent and "
            "tf_band_power_reduced, so they are guaranteed-duplicate scores.",
            "Include all 17 Phase-1 extractors: rejected as redundant compute."],
        rationale="Cover time, frequency, time-frequency and spatial families without "
                  "scoring provably-duplicate representations.",
        consequences="Latency/layer analysis adds a time-resolved view the fixed "
                     "feature sets cannot provide.", tags=T + ["eeg"])

    log_decision(
        agent=A, title="Cross-validation scheme",
        choice="KFold(n_splits=5, shuffle=True, random_state=1234) on trials within a "
               "subject, applied identically to encoding, decoding, retrieval, CCA.",
        alternatives=[
            "GroupKFold by video: unnecessary — each video appears exactly once per "
            "subject (no within-subject repeats, data fact), so no video's trials can "
            "straddle a fold under any split. The 'don't split a video across folds' "
            "rule is therefore trivially satisfied.",
            "LeaveOneOut: too expensive for 960+ trials x many pairs."],
        rationale="Standard, reproducible, and the single-trial-per-video structure "
                  "removes the usual leakage concern.",
        consequences="Reported scores are cross-validated, not in-sample.", tags=T)

    log_decision(
        agent=A, title="Encoding PCA fitted on training-fold videos only",
        choice="Stimulus embeddings are PCA-reduced (<=100 comps, capped at "
               "n_train_fold-1 and ambient dim) with the PCA FITTED INSIDE each fold on "
               "training rows, then applied to held-out rows. Ridge alpha chosen by "
               "RidgeCV over logspace(-1,6,15).",
        alternatives=[
            "Fit PCA on all videos then CV: rejected — leaks held-out structure into the "
            "projection.",
            "No PCA (raw 139968-dim AlexNet): rejected — p>>n, unstable ridge."],
        rationale="Keeps every fitted object train-only and stabilises p>>n regression.",
        consequences="Scores depend on the PCA cap; recorded in each result's config.",
        tags=T + ["encoding"])

    log_decision(
        agent=A, title="Decoding EEG PCA cap and target standardisation",
        choice="EEG features PCA-reduced to <=200 comps inside each fold; embedding "
               "targets z-scored with training-fold statistics; RidgeCV alphas "
               "logspace(-1,6,15). Retrieval scored by cosine, which is scale-robust.",
        alternatives=["Partial least squares: comparable but adds a hyper-parameter; "
                      "ridge+PCA is the Phase-1-consistent choice."],
        rationale="Mirrors the encoding regulariser; cosine retrieval is invariant to "
                  "the target scale shift between train and test.",
        consequences="200-comp EEG cap can under-fit very high-dim feature sets; "
                     "recorded in config.", tags=T + ["decoding"])

    log_decision(
        agent=A, title="Retrieval candidate pool and repeats",
        choice="Pool size 102 (matches the final task); for each held-out trial the "
               "pool is its true item + 101 distractors drawn without replacement from "
               "other trials; the whole sweep repeats 20 times with fixed seeds. "
               "Both directions reported (EEG->stim and stim->EEG). Headline carries "
               "top-1, top-5 AND median rank with 95% Wilson CI and one-sided binomial "
               "p vs 1/102 and 5/102.",
        alternatives=[
            "Pool = all held-out videos: easier but does not match the 102-way final "
            "task.",
            "Single draw: noisier; 20 repeats stabilise the estimate.",
            "Top-1 only: rejected — Phase 1's null came from top-1 alone; top-1 is the "
            "least sensitive metric here (baseline top-1 n.s. but top-5 significant)."],
        rationale="Matches the real evaluation and avoids the top-1 blind spot.",
        consequences="When a partial embedding has <102 common videos the pool shrinks "
                     "and chance rises; recorded per result.", tags=T + ["retrieval"])

    log_decision(
        agent=A, title="RDM distance metrics and significance",
        choice="EEG RDM = 1 - Pearson correlation across trials; stimulus RDM = "
               "1 - cosine similarity. Second-order similarity = Spearman rho on the "
               "upper triangles. Significance = video-label permutation test with 1000 "
               "shuffles (300 in --quick); uncertainty = 1000-sample bootstrap CI over "
               "videos.",
        alternatives=[
            "Euclidean EEG RDM: dominated by amplitude, which is confounded by the "
            "train/test std mismatch, so correlation distance is preferred.",
            "Parametric p for rho: rejected — RDM cells are not independent, so a "
            "permutation null is the correct reference."],
        rationale="Correlation/cosine distances are scale-robust; the permutation null "
                  "respects RDM dependence.",
        consequences="Permutation p is granular to 1/(n_perm+1).", tags=T + ["rsa"])

    log_decision(
        agent=A, title="Latency: 100 ms non-overlapping EEG windows",
        choice="The preprocessed (n,17,750) @250 Hz array is sliced into 30 "
               "non-overlapping 100 ms windows (25 samples each); each (window, layer) "
               "cell is the mean cross-validated encoding r.",
        alternatives=[
            "50 ms overlapping windows: finer but 12-13 samples per window is too few "
            "for stable per-window encoding and inflates the number of comparisons.",
            "Reuse the Phase-1 ERP windows: uneven widths complicate the heatmap axis."],
        rationale="100 ms matches the frame grid of the time-resolved embeddings and "
                  "gives enough samples per window.",
        consequences="Cannot resolve sub-100 ms latency differences.", tags=T + ["latency"])

    log_decision(
        agent=A, title="CKA and CCA definitions and regularisation",
        choice="Linear centered CKA (HSIC form) as the representational-similarity "
               "index in [0,1]; regularized CCA via per-view PCA truncation to 50 comps "
               "(30 in --quick) inside each fold, scored by held-out mean canonical "
               "correlation over 5 components.",
        alternatives=[
            "RBF CKA: adds a bandwidth hyper-parameter; linear CKA is the standard "
            "baseline and matches linear encoding/decoding.",
            "Unregularized CCA: singular for p>>n; PCA truncation is the standard fix."],
        rationale="Linear everything keeps the comparison coherent; held-out CCA avoids "
                  "the in-sample upper bound.",
        consequences="Linear CKA misses purely nonlinear shared structure.",
        tags=T + ["cka", "cca"])

    log_decision(
        agent=A, title="Uniqueness: redundancy threshold and incremental gain",
        choice="Flag an embedding pair as redundant if linear CKA > 0.8 AND adding one "
               "to the other yields no positive, non-zero-straddling fold-wise encoding "
               "gain (paired t over folds, 95% CI). CKA computed on the video IDs COMMON "
               "to all well-powered embeddings so every cell compares the same videos. "
               "Hierarchical clustering uses average linkage on 1 - CKA.",
        alternatives=[
            "CKA > 0.9: too lenient; 0.8 is the common redundancy convention.",
            "Gain by a single train/test split: noisier than paired fold differences."],
        rationale="High similarity alone is not redundancy; requiring no added "
                  "predictive value is the operational test. CIs that straddle 0 are "
                  "never called an improvement.",
        consequences="Threshold is a convention; all pairwise CKA + gains are stored so "
                     "the flag can be revisited.", tags=T + ["uniqueness"])

    log_decision(
        agent=A, title="Calibration controls and their pass criteria (DONE FIRST)",
        choice="(a) shuffled-video-label null, (b) Gaussian random-vector embeddings, "
               "(c) synthetic positive control: fake EEG = W @ embedding + Gaussian "
               "noise at SNRs {inf,4,1,0.25,0.0625,0}. PASS requires: random+shuffled "
               "collapse to chance; and the SNR sweep recovers retrieval clearly above "
               "chance at high SNR, decreases monotonically (2% tolerance), and "
               "collapses to chance at SNR 0.",
        alternatives=[
            "Only a shuffled null: rejected — it cannot show the pipeline RECOVERS "
            "signal, only that it is not trivially broken. The SNR sweep proves "
            "discrimination.",
            "Fixed SNR grid without the inf/0 anchors: rejected — the anchors make "
            "'recovers' and 'collapses' unambiguous."],
        rationale="These controls are the evidence the whole suite discriminates and "
                  "they let the suite be validated before any real embedding exists.",
        consequences="If any control fails, no real result is trusted; the runner "
                     "records all_passed and every pair's result file carries the "
                     "control summary.", tags=T + ["controls"])

    log_decision(
        agent=A, title="Significance threshold and multiple-comparison correction",
        choice="One-sided binomial p vs chance for every headline top-1/top-5, pooled "
               "across subjects, with 95% Wilson CIs. Multiple embeddings are corrected "
               "by Bonferroni over the number of embeddings scored in the run; a result "
               "is called significant only if p < 0.05 / n_embeddings. Differences whose "
               "95% CIs overlap are never described as improvements.",
        alternatives=[
            "No correction: rejected — many embeddings are compared.",
            "FDR (Benjamini-Hochberg): less conservative; Bonferroni is reported as the "
            "strict bar, matching the Phase-1 baseline's Bonferroni/17 framing."],
        rationale="Conservative and consistent with the baseline reference row.",
        consequences="Bonferroni over many embeddings is strict; raw p and CIs are also "
                     "reported so a reader can apply a different correction.",
        tags=T + ["stats"])

    log_decision(
        agent=A, title="Device policy and the MPS-vs-CPU gate",
        choice="Default compute is CPU/numpy, which is the reported value for every "
               "statistic. Linear CKA additionally runs on MPS when available and the "
               "MPS value is gated against the CPU value with assert_close_to_cpu "
               "(rtol 1e-4, atol 1e-6); the gate result is stored in each result JSON. "
               "The reported CKA is always the CPU number.",
        alternatives=[
            "Report the MPS number directly: rejected — a float32 GPU reduction can "
            "shift a statistic; the gate is a correctness check, not a permission check.",
            "CPU only, never touch MPS: allowed, but the gate lets heavy kernels use "
            "MPS safely when they agree."],
        rationale="Honours the hard rule that any reported statistic computed on MPS "
                  "must pass a CPU-equivalence gate.",
        consequences="If a future heavy kernel fails the gate it must fall back to CPU; "
                     "the stored gate makes that auditable.",
        tags=T + ["mps", "stats"])

    log_decision(
        agent=A, title="Final-dev validation: direction, candidate ranking, both norms",
        choice="Decoding (EEG->embedding) only, sub-01..04. Fit everything on train, "
               "predict a test embedding per test trial, cosine-rank the 102 cached "
               "test-video embeddings, score vs tst_stim. Report BOTH without and with "
               "test-split-only EEG feature normalisation because test std is 62-68% of "
               "train std. pipeline_mutated is DERIVED from a before/after deep hash of "
               "every fitted object.",
        alternatives=[
            "Encoding direction for the final: rejected — Phase 1 diagnosed EEG->stimulus "
            "retrieval as the task-relevant direction.",
            "Correct the amplitude mismatch by fitting on test: rejected — leakage."],
        rationale="Matches the real 102-way task and surfaces the amplitude confound "
                  "without correcting it on fitted objects.",
        consequences="test_split_norm uses test statistics to rescale test rows only "
                     "(no fitted object sees test), reported as a sensitivity row.",
        tags=T + ["final", "leakage"])

    print("V-C decisions logged.")


if __name__ == "__main__":
    main()
