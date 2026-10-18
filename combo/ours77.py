"""ours77: equal-weight rrf+hub mean of our 77 feature scorers (15 winfuse windows x V-JEPA-G, V-JEPA-L, S3D, CLAP, SigLIP
decoding + 2 encoders), fitted on one person's training recordings, scored on that person's test recordings.
    python combo/ours77.py OUT sub-05          -> OUT/sub-05_ours77.npy   (local; same as modal_combo.py --o77)
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combo  # noqa: E402

if __name__ == '__main__':
    out, sub = sys.argv[1], sys.argv[2]
    fz = {'channels': combo.NEW_SPECS, 'transform': 'rrf+hub', 'weights': [1 / len(combo.NEW_SPECS)] * len(combo.NEW_SPECS)}
    os.makedirs(out, exist_ok=True); p = os.path.join(out, '_ours77.json'); json.dump(fz, open(p, 'w'))
    combo.predict(out, sub, int(sub[-1]), frozen=p)
    os.replace(os.path.join(out, f'{sub}_combo_scores.npy'), os.path.join(out, f'{sub}_ours77.npy'))
    print('ours77 written', sub)
