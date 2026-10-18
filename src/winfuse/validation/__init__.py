"""Agent C — generic validation suite for EEG feature extractors.

This package scores ANY extractor that implements the frozen `BaseExtractor`
contract (`src.common.feature_api`) without knowing its internals. It is built so
that an extractor that does not yet exist, or that raises, degrades gracefully:
the runner records the error for that method and moves on.

Modules
-------
blink              metric 1 — blink/eye-movement contamination (EOG proxy)
erp                metric 2 — evoked response quality (grand-average ERP)
reliability        metric 3 — split-half + cross-subject consistency
descriptors        metric 4a — cheap cached video/audio descriptors
retrieval          metric 4b — ridge EEG→descriptor retrieval proxy + null
final_dev_validation  metric 5 — the ONLY place tst/ is read (sub-01..04)
stubs              synthetic extractors with known properties, used to prove the
                   metrics actually discriminate
run_suite          CLI entry point that ties it all together

All fitting happens on trn/ only. tst/ is touched exclusively in
`final_dev_validation.py`, for dev subjects, read-only.
"""
