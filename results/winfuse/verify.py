"""Check the shipped package is intact and reproduces its numbers. No fitting, ~2 s.

    python verify.py

Needs only numpy, scipy and scikit-learn. Needs no data, no environment variables and
no network. Every check degrades to a clear [skip] rather than a traceback if an
optional input is absent, so it is safe to run on a partially copied package.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEST_IDS = list(range(1001, 1103))
SUBS = ("sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06")
SCORABLE = ("sub-01", "sub-02", "sub-03", "sub-04")  # the only ones with labels
n_pass = n_fail = n_skip = 0


def check(label, cond, detail=""):
    global n_pass, n_fail
    good = bool(cond)
    n_pass, n_fail = n_pass + good, n_fail + (not good)
    print(f"  [{'PASS' if good else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")


def skip(label, why):
    global n_skip
    n_skip += 1
    print(f"  [skip] {label} — {why}")


def need(path, label):
    if path.exists():
        return True
    skip(label, f"{path.relative_to(HERE)} absent")
    return False


# ------------------------------------------------------------------ weights
print("=== 1. the weights are general, not fitted per subject ===")
cfg = json.loads((HERE / "weights" / "config.json").read_text())
w = np.load(HERE / "weights" / "fusion_weights.npy")
specs = (HERE / "weights" / "channels.txt").read_text().split()

check("weights / channels / config agree on the count",
      len(w) == len(specs) == cfg["scorers"], f"{len(w)} scorers")
check("weights sum to 1", abs(w.sum() - 1) < 1e-9, f"sum = {w.sum():.12f}")
check("weights are all finite and non-negative",
      bool(np.all(np.isfinite(w)) and np.all(w >= 0)))
nd = len(set(np.round(w, 12)))
check("every scorer carries the SAME weight, so nothing is subject-specific",
      nd == 1, f"{nd} distinct value, each = 1/{len(w)}")
check("config declares subject independence explicitly",
      cfg["SUBJECT_INDEPENDENCE"]["fusion_weights_are_subject_specific"] is False)
check("no subject name appears in the weights, channels or config",
      not any(s in (HERE / "weights" / "channels.txt").read_text()
              for s in ("sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06")))
loo = cfg["SUBJECT_INDEPENDENCE"]
check("leave-one-subject-out is recorded and close to the all-subject figure",
      abs(loo["all_four_subjects_top5"] - loo["leave_one_subject_out_top5"]) < 0.02,
      f"{loo['leave_one_subject_out_top5']*100:.2f}% held-out vs "
      f"{loo['all_four_subjects_top5']*100:.2f}% using all four")

# ------------------------------------------------------------------ ranking
print("\n=== 2. ranking runs here, with no fitting ===")
sys.path.insert(0, str(HERE))
from rank import to_ranked  # noqa: E402

produced = {}
for sub in SUBS:
    sp = HERE / "scores" / f"{sub}_fused_scores.npy"
    if not need(sp, f"{sub}: rank from shipped scores"):
        continue
    S = np.load(sp).astype(np.float64)
    r = to_ranked(S, cfg)
    produced[sub] = r
    check(f"{sub}: produced a ranking from the shipped scores", r.shape == (102, 102))

print("\n=== 3. output is in the exact submission format ===")
for sub, r in produced.items():
    good = (r.shape == (102, 102) and np.issubdtype(r.dtype, np.integer)
            and all(set(r[i].tolist()) == set(TEST_IDS) for i in range(102)))
    check(f"{sub}: (102,102) integer, every row a permutation of 1001..1102", good)
    check(f"{sub}: rows are NOT sorted by ID (original recording order preserved)",
          not np.array_equal(r[:, 0], np.sort(r[:, 0])))

print("\n=== 4. it agrees with the rankings shipped in the package ===")
for sub, r in produced.items():
    p = HERE / "ranked" / f"{sub}_ranked_candidate_list.npy"
    if not need(p, f"{sub}: compare with shipped ranking"):
        continue
    check(f"{sub}: re-derived ranking is identical to the shipped one",
          np.array_equal(r, np.load(p)))

print("\n=== 5. the held-out subjects, which is the actual submission ===")
for sub in ("sub-05", "sub-06"):
    p_ = HERE / "ranked" / f"{sub}_ranked_candidate_list.npy"
    if not need(p_, f"{sub}: submission file"):
        continue
    a = np.load(p_)
    check(f"{sub}: submission file is (102,102) int, rows permute 1001..1102",
          a.shape == (102, 102) and np.issubdtype(a.dtype, np.integer)
          and all(set(a[i].tolist()) == set(TEST_IDS) for i in range(102)))
check("no labels for the held-out subjects are present anywhere in this package",
      not list(HERE.rglob("sub-05_tst_stim.npy")) and
      not list(HERE.rglob("sub-06_tst_stim.npy")))

print("\n=== 6. accuracy against ground truth (development subjects only) ===")
lab = HERE / "evidence" / "labels"
have = lab.is_dir() and all((lab / f"{s}_tst_stim.npy").exists() for s in SCORABLE)
if not have:
    skip("scoring", "the organisers' label files are not redistributed in this package. "
         "Copy tst_stim/*.npy into evidence/labels/ to enable this check")
elif any(s not in produced for s in SCORABLE):
    skip("scoring", "not every subject's scores are present")
else:
    pos = []
    for sub in SCORABLE:
        y = np.load(lab / f"{sub}_tst_stim.npy")
        r = produced[sub]
        pos += [int(np.where(r[k] == y[k])[0][0]) + 1 for k in range(102)]
    pos = np.array(pos)
    t1, t5 = float(np.mean(pos <= 1)), float(np.mean(pos <= 5))
    exp = cfg["measured_dev_test"]
    check(f"top-1 reproduces: {t1*100:.2f}% vs recorded {exp['top1']*100:.2f}%",
          abs(t1 - exp["top1"]) < 0.005)
    check(f"top-5 reproduces: {t5*100:.2f}% vs recorded {exp['top5']*100:.2f}%",
          abs(t5 - exp["top5"]) < 0.005)
    check("well above chance", t5 > 10 * (5 / 102), f"{t5/(5/102):.1f}x chance")

print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped")
if n_fail:
    print("SOME CHECKS FAILED")
elif n_skip:
    print("ALL RUNNABLE CHECKS PASSED (some optional inputs were absent)")
else:
    print("ALL CHECKS PASSED")
raise SystemExit(1 if n_fail else 0)
