"""Export each model's fused score matrix, *before* post-processing, for the NeuroBytes app.

The browser then runs the post-processing itself (CSLS k=2 and Sinkhorn 3 / 0.5,
ported line-for-line in src/lib/rank.ts) and the app checks the result against the
rankings computed here by the team's own code.

    python app/scripts/export_models.py --team-repo /path/to/neurobytes \
        [--out results/neurobytes_models]

C13, C10 and WF come from the final submission repo (Sushruths04/neurobytes):
    --team-repo /path/to/neurobytes --models C13,C10,WF
C11 and C12 need the earlier neurodecode-mindread repo (origin/main >= b6377ab), which
still ships combo/c11_inputs and c12_inputs: --models C12,C11. Other models in an
existing manifest are left untouched. Needs numpy + scikit-learn.
Writes, per model and subject:
    <out>/<model>/<sub>_pre.npy       float64 (102, 102)  fused scores, pre post-processing
    <out>/<model>/<sub>_ranked.npy    int64   (102, 102)  team code's final ranking (for checking)
    <out>/manifest.json               models, formulas, post-processing, dev numbers
Nothing here is committed: <out> sits under results/, which is gitignored.
No ground-truth labels are read or written; dev accuracy is computed by the app.
"""
import argparse, json, os, sys

import numpy as np

SUBS = ("sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06")

# Each model: how to build the pre-post-processing matrix, and which post-processing follows.
# Formulas and weights are copied from the team's frozen configs (configs/frozen_C1x.json).
MODELS = {
    # final C13 (configs/frozen_C13.json, 18 Oct): networks h6 and h10, organiser data only
    "C13": {"families": [("h10", 0.075), ("h6", 0.075)], "nc_dir": "c13_inputs", "csls": 2,
            "formula": "0.85 * C10 blend + 0.075 * rrf+hub(network h6) + 0.075 * rrf+hub(network h10)"},
    "C12": {"families": [("h7", 0.15)], "nc_dir": "c12_inputs", "csls": 2,
            "formula": "0.85 * C10 blend + 0.15 * rrf+hub(network h7)"},
    "C11": {"families": [("C11conv", 0.1)], "nc_dir": "c11_inputs", "csls": 2,
            "formula": "0.9 * C10 blend + 0.1 * rrf+hub(network conv)"},
    "C10": {"families": [], "nc_dir": None, "csls": 2,
            "formula": "0.85 * ((509/586) winfuse + (77/586) mindread-77) + 0.15 * rrf+hub(T2)"},
    "WF": {"families": None, "nc_dir": None, "csls": None,
           "formula": "winfuse submission2: 509 ridge scorers, rrf+hub, equal weights"},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team-repo", required=True)
    ap.add_argument("--models", default=",".join(MODELS), help="comma list; others are left untouched")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "neurobytes_models"))
    a = ap.parse_args()
    team = os.path.abspath(a.team_repo)
    out = os.path.abspath(a.out)
    sys.path.insert(0, os.path.join(team, "combo"))
    import pickle

    import c10  # noqa: E402  (also puts the team's src/ on sys.path for csls)
    import c11  # noqa: E402

    def t2_test(sub):
        if sub in ("sub-05", "sub-06"):
            return np.load(f"{c10.T2}/T2_test_{sub}.npy")
        return pickle.load(open(f"{c10.T2}/{sub}__T2_mindread_enc.pkl", "rb"))["test"]

    def pre(name, sub):
        m = MODELS[name]
        if m["families"] is None:  # winfuse alone
            return np.load(f"{c10.WF}/{sub}_fused_scores.npy").astype(np.float64)
        c11.NC_DIR = os.path.join(team, "combo", m["nc_dir"]) if m["nc_dir"] else c11.NC_DIR
        w = sum(b for _, b in m["families"])
        if not m["families"]:  # C10: blend is the whole pre matrix
            return c11.blend(sub, t2_test(sub))
        return (1 - w) * c11.blend(sub, t2_test(sub)) + sum(b * c10.rrf_hub(c11.nc(sub, t)) for t, b in m["families"])

    def post(name, S):
        if MODELS[name]["csls"]:
            S = c10.csls(S, MODELS[name]["csls"])
        return c10.sinkhorn(S, c10.SK_ITERS, c10.SK_TAU)

    mf = os.path.join(out, "manifest.json")
    manifest = json.load(open(mf)) if os.path.exists(mf) else {"models": {}}
    manifest.update({"sinkhorn": {"iters": int(c10.SK_ITERS), "tau": float(c10.SK_TAU)}})
    for name in a.models.split(","):
        m = MODELS[name]
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        subs = []
        for sub in SUBS:
            try:
                S = pre(name, sub)
            except FileNotFoundError as e:
                print(f"  {name} {sub}: skipped ({e.filename} missing)")
                continue
            assert S.shape == (102, 102), S.shape
            r = c10.TEST_IDS[np.argsort(-post(name, S), 1, kind="stable")].astype(np.int64)
            np.save(f"{d}/{sub}_pre.npy", S.astype(np.float64))
            np.save(f"{d}/{sub}_ranked.npy", r)
            subs.append(sub)
        manifest["models"][name] = {"formula": m["formula"], "csls_k": m["csls"], "subjects": subs, "source": f"team repo {team}"}
        print(f"{name}: {len(subs)} subjects -> {d}")

    # Cross-check against the team's shipped submissions where they exist.
    for name in [n for n in ("C13", "C12") if n in a.models.split(",")]:
        for sub in ("sub-05", "sub-06"):
            f = os.path.join(team, "submissions", name, f"{sub}_ranked_candidate_list.npy")
            if os.path.exists(f):
                same = np.array_equal(np.load(f), np.load(os.path.join(out, name, f"{sub}_ranked.npy")))
                print(f"  {name} {sub} vs submissions/{name}: {'identical' if same else 'DIFFERENT'}")
                assert same
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print("wrote", os.path.join(out, "manifest.json"))


if __name__ == "__main__":
    main()
