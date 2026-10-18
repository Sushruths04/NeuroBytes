#!/bin/bash
# Quickstart: reproduce C13 (dev test 36.03% top-1 / 67.16% top-5) and the sub-05/06 submission in a few minutes.
# Uses the released network weights + the shipped component scores; recomputes the networks from your raw test EEG.
#   DATA=/path/to/eeg-mind-reading-challenge bash scripts/reproduce_c13.sh
# Optional: WEIGHTS=/path/to/c13o_weights_20261018 (skip the download).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
: "${DATA:?set DATA to the official challenge folder (it must contain tst/sub-0X_tst.npy)}"
export C13_ALIGN_TO_DATA=1 WINFUSE_DATA_ROOT="$DATA" PYTHONPATH="$REPO/src:${PYTHONPATH:-}" TMPDIR="${TMPDIR:-/tmp}"
WORK=${WORK:-$REPO/reproduce}; mkdir -p "$WORK"
if [[ -z "${WEIGHTS:-}" ]]; then
  WEIGHTS="$WORK/c13o_weights_20261018"
  if [[ ! -d "$WEIGHTS" ]]; then
    echo "[1/4] downloading weights (release v1.0, 161 MB)"
    gh release download v1.0 -R Sushruths04/neurobytes -p 'c13o_weights_20261018.tar.gz' -D "$WORK" --skip-existing
    echo "c81efde41c2ee259160e48c5242d215b1673cc3a6233455ae422937679a3ab3f  $WORK/c13o_weights_20261018.tar.gz" | sha256sum -c -
    tar -xzf "$WORK/c13o_weights_20261018.tar.gz" -C "$WORK"
  fi
fi
echo "[2/4] scoring raw test EEG with the two networks (h6, h10; 5 seeds each)"
for t in h6 h10; do for s in 1 2 3 4 5 6; do
  python combo/nc_predict.py --weights "$WEIGHTS" --tag $t --subject sub-0$s --eeg "$DATA/tst/sub-0${s}_tst.npy" --out "$WORK/nets"
done; done
echo "[3/4] C13 on the development test (sub-01..04); expected: top1 36.03 top5 67.16 rank 5.99 (GPU) / 6.00 (CPU)"
NC_DIR="$WORK/nets" NC_FAMILIES=h10:0.075,h6:0.075 python combo/c11.py --dev
echo "[4/4] C13 submission files for sub-05/06"
NC_DIR="$WORK/nets" NC_FAMILIES=h10:0.075,h6:0.075 python combo/c11.py --out "$WORK/submission"
NC_DIR="$WORK/nets" NC_FAMILIES=h10:0.075,h6:0.075 python combo/c11.py --check "$WORK/submission"
