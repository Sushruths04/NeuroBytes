# `results/<method>/metrics.json` schema (version 1.0)

Written only by `src.common.results.ResultsWriter`. `summary.md` in the same
directory is rendered from exactly this dict, so the two can never disagree.

Any key may be absent. A missing key renders as `–` in tables and **must not** be
filled in by hand in a report — if a metric is absent, the report says it is
absent and `notes` explains why.

## Top level

| key | type | meaning |
|---|---|---|
| `schema_version` | str | `"1.0"` |
| `method` | str | extractor `name`; equals the directory name |
| `agent` | str | `"A"`, `"B"` or `"C"` |
| `generated` | str | ISO 8601 with timezone offset |
| `environment` | object | python/platform/git rev + library versions, auto-filled |
| `meta` | object | method-level info (below) |
| `subjects` | object | `"sub-01"` → per-subject metric object (below) |
| `aggregate` | object | cross-subject summaries; free-form flat key→value |
| `notes` | array of str | caveats, limitations, "could not be computed because …" |
| `figures` | array of str | repo-relative PNG paths produced for this method |

## `meta`

| key | type | meaning |
|---|---|---|
| `description` | str | one line, used as the report subtitle |
| `n_features` | int | columns returned by `transform` |
| `params` | object | the extractor's `params`, verbatim |
| `requires_raw` | bool | true if the method consumes 500 Hz unfiltered epochs |
| `preprocessing` | object | `PreprocConfig` as a dict, for provenance |
| `feature_names_head` | array of str | first ~20 feature names, for sanity-checking |

## `subjects["sub-0X"]`

Validation suite output. Group keys are optional; include the ones you computed.

### `runtime_s`, `peak_rss_mb`, `n_features`

| key | type | meaning |
|---|---|---|
| `runtime_s.fit` | float | seconds in `fit` |
| `runtime_s.transform` | float | seconds in `transform` on the training set |
| `runtime_s.total` | float | seconds for the whole per-subject run |
| `peak_rss_mb` | float | peak resident set size of the worker, MB |
| `n_trials` | int | training trials for this subject |
| `n_features` | int | same as `meta.n_features`, repeated for table convenience |

### `blink` — metric 1, contamination

| key | type | meaning |
|---|---|---|
| `proxy_channels` | array of int | channel **indices** used as the EOG proxy |
| `proxy_channel_scores` | array of float | the ranking statistic for all 17 channels |
| `threshold_desc` | str | the detection rule in words, including the numeric threshold |
| `threshold_value` | float | the numeric threshold actually applied |
| `n_trials_with_blink` | int | |
| `frac_trials_with_blink` | float | |
| `per_feature_auc` | object | feature name → AUC of that feature vs. the blink flag |
| `mean_abs_auc` | float | mean of \|AUC − 0.5\| + 0.5 over features; **lower is cleaner** |
| `max_abs_auc` | float | worst single feature; **lower is cleaner** |
| `top_blink_features` | array | the 10 most blink-sensitive feature names with their AUC |
| `after_ica` | object | same sub-keys, recomputed after train-fitted ICA |
| `after_rejection` | object | same sub-keys, recomputed after dropping blink trials |

### `erp` — metric 2, evoked response quality

| key | type | meaning |
|---|---|---|
| `peak_latency_s` | float | latency of the largest early-window deflection in the grand average |
| `peak_amplitude_au` | float | its amplitude, **arbitrary units** (the data is not in µV) |
| `peak_channel` | int | channel index carrying it |
| `snr` | float | peak amplitude ÷ across-trial std at the peak |
| `onset_response_present` | bool | whether a deflection exceeding the documented criterion exists in 100–200 ms |
| `onset_criterion` | str | the criterion, in words with numbers |
| `channels_carrying_onset` | array of int | channel indices above criterion |
| `per_window` | object | ERP window name → `{mean, peak, latency_s}` |
| `auditory` | object | audio-envelope-related response, or `{"status": "..."}` if not computable |

### `reliability` — metric 3

| key | type | meaning |
|---|---|---|
| `split_half_r` | float | Pearson/Spearman r between odd- and even-trial feature means |
| `spearman_brown` | float | `2r/(1+r)`, the corrected value |
| `per_feature_split_half` | object | feature name → r |
| `n_odd`, `n_even` | int | trial counts in each half |
| `cross_subject_consistency` | object | subject pair → r of grand-average time courses |

### `retrieval_proxy` — metric 4, stimulus specificity

| key | type | meaning |
|---|---|---|
| `descriptors` | array of str | descriptor names actually used |
| `pearson_r` | object | descriptor → out-of-fold Pearson r |
| `top1`, `top5` | float | retrieval accuracy among held-out **training** videos |
| `top1_null`, `top5_null` | float | shuffled-label null baseline |
| `median_rank` | float | |
| `n_candidates` | int | pool size the ranks are computed against |
| `chance_top1`, `chance_top5` | float | `1/n_candidates`, `5/n_candidates` |
| `cv` | str | the CV scheme, e.g. `"KFold(5, shuffle=True, seed=1234)"` |
| `ridge_alphas` | array of float | the alpha grid searched |

### `final_dev` — metric 5, the only place `tst/` is used

| key | type | meaning |
|---|---|---|
| `top1`, `top5` | float | against `tst_stim/`, sub-01..04 only |
| `median_rank` | float | |
| `n_candidates` | int | 102 |
| `chance_top1`, `chance_top5` | float | `1/102 ≈ 0.0098`, `5/102 ≈ 0.049` |
| `pipeline_mutated` | bool | **must be `false`**; asserted by the final-validation script |

## Conventions

- Floats that are NaN or Inf are serialised as `null`, never as the string `"NaN"`.
- Amplitudes are in **arbitrary units**. The data is not in µV
  (`reports/00_data_inspection.md` §6).
- Channels are **integer indices 0–16**. There are no channel names
  (`reports/00_data_inspection.md` §7).
- "Blink" numbers are **proxy** numbers: there is no EOG channel, so the proxy
  channels are chosen by the documented ranking statistic and recorded in
  `blink.proxy_channels`.
- Every `*_null` field is a shuffled-label baseline computed with the same code
  path as the real metric, so chance level is always visible next to the result.
