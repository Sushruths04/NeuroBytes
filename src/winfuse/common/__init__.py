"""Shared contracts for the EEG feature benchmark.

FROZEN: sub-agents must not edit anything in `src/common/`. Request changes via
the orchestrator.

Modules
-------
config          paths, sampling rates, bands, ERP windows, seed, worker cap
io              load_subject_train / load_subject_test / load_test_labels (+ leakage guards)
preprocess      PreprocConfig, Preprocessor, preprocessed_train / preprocessed_test
feature_api     BaseExtractor, @register, get_extractor, list_extractors
results         ResultsWriter (writes metrics.json + summary.md)
decisions       log_decision
"""
from . import config, decisions, feature_api, io, preprocess, results  # noqa: F401

__all__ = ["config", "io", "preprocess", "feature_api", "results", "decisions"]
