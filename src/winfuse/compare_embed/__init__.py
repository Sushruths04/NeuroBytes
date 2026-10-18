"""Agent V-C: EEG-vs-stimulus comparison suite (Phase 2).

Compares Phase-1 EEG feature sets against stimulus embeddings (produced by agents
V-A/V-B, or synthetic controls built here) in a shared space, using encoding,
decoding, retrieval, RSA, latency/layer, CKA/CCA and uniqueness analyses, with a
calibration-control battery that must pass before any real result is trusted.

Scope rules (see the task brief):
  * owns ONLY src/compare_embed/, results/compare_embed/, figures/compare_embed/,
    decisions/V-C.md (via the shared logger) and reports/VC_comparison_suite.md
  * reads frozen contracts in src/common/ but never edits them
  * all fitting on training data only; tst/ + tst_stim/ touched ONLY in
    final_dev_validation.py, sub-01..04; sub-05/06 never touched
  * every reported statistic computed on MPS must pass assert_close_to_cpu
"""
from __future__ import annotations

__all__ = ["AGENT", "RESULTS_SUBDIR", "FIGURES_SUBDIR"]

AGENT = "V-C"
RESULTS_SUBDIR = "compare_embed"
FIGURES_SUBDIR = "compare_embed"
