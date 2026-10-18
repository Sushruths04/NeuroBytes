#!/bin/bash
# End-to-end C13 pipeline (native). Usage (from the repository root):
#   DATA=/path/to/eeg-mind-reading-challenge bash scripts/run_c13_pipeline.sh [features|scorers|networks|assemble|all]
# Stages:
#   features : download the v1.0 release (our video features + winfuse embedding banks): 2.8 GB download, about 6 GB of free disk
#   scorers  : rebuild the ridge components for all six people (winfuse aligned 509, ours77, T2)
#   networks : retrain the two contrastive networks (deterministic) and save weights  [or download release v1.0]
#   assemble : C13 = fusion + CSLS + Sinkhorn -> submissions/C13_rebuilt/sub-0{5,6}_ranked_candidate_list.npy
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
: "${DATA:?set DATA to the official challenge folder (contains trn/, tst/)}"
export WINFUSE_DATA_ROOT="$DATA" PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
OUT=${OUT:-$REPO/rebuild}; mkdir -p "$OUT"
STAGE=${1:-all}
H6=clip_l14,vjepa2_g_l30,dinov2_b_l4_8_12,wf:alexnet_layers@layer_fc6,wf:resnet50@layer_layer3,wf:videomae_base
H10=$H6,wf:alexnet_layers@layer_conv5,wf:whisper_base,wf:hc_visual,wf:xclip_base@layer_mit_video

if [[ $STAGE == features || $STAGE == all ]]; then
  for f in c13o_features.sha256 c13o_features_mindread.tar c13_features_winfuse.tar.part-aa c13_features_winfuse.tar.part-ab; do
    if command -v gh >/dev/null; then gh release download v1.0 -R Sushruths04/neurobytes -p "$f" -D "$OUT" --skip-existing
    else [[ -f "$OUT/$f" ]] || curl -fL -o "$OUT/$f" "https://github.com/Sushruths04/neurobytes/releases/download/v1.0/$f"; fi
  done
  (cd "$OUT" && sha256sum -c c13o_features.sha256)
  mkdir -p outputs cache && tar -xf "$OUT/c13o_features_mindread.tar" -C outputs
  cat "$OUT"/c13_features_winfuse.tar.part-* | tar -x -C cache
fi
if [[ $STAGE == scorers || $STAGE == all ]]; then
  mkdir -p results/compare_embed && cp results/winfuse/evidence/_frozen_pipeline.json results/compare_embed/
  # cross_subject caches its output in cache/align and reuses it; FORCE=1 rebuilds
  [[ ${FORCE:-0} == 1 ]] && rm -f cache/align/sub-0?_test_aligned.npy
  python -m winfuse.compare_embed.cross_subject --test --subjects sub-01,sub-02,sub-03,sub-04
  python -m winfuse.compare_embed.cross_subject --test --subjects sub-05,sub-06
  mkdir -p "$OUT/wf"; for s in 1 2 3 4 5 6; do cp cache/align/sub-0${s}_test_aligned.npy "$OUT/wf/sub-0${s}_fused_scores.npy"; done
  for s in 1 2 3 4 5 6; do python combo/ours77.py "$OUT/inputs" sub-0$s; python combo/t2_scores.py "$OUT/inputs" $s; done
fi
if [[ $STAGE == networks || $STAGE == all ]]; then
  NC_TAG=h6  NC_T=150 NC_SEEDS=0,1,2,3,4 NC_TARGETS=$H6  NC_SAVE="$OUT/weights" NC_OUT=rebuild_nets python combo/neural_clip.py final
  NC_TAG=h10 NC_T=150 NC_SEEDS=0,1,2,3,4 NC_TARGETS=$H10 NC_SAVE="$OUT/weights" NC_OUT=rebuild_nets python combo/neural_clip.py final
  for t in h6 h10; do NC_TARGETS=$([ $t = h6 ] && echo $H6 || echo $H10) python combo/nc_candidates_fp32.py "$OUT/weights" $t; done
  mkdir -p "$OUT/weights/preproc"; for s in 1 2 3 4 5 6; do cp cache/preproc/sub-0${s}_prep_*.joblib "$OUT/weights/preproc/sub-0${s}_prep.joblib"; done
fi
if [[ $STAGE == assemble || $STAGE == all ]]; then
  W=${WEIGHTS:-$OUT/weights}; mkdir -p "$OUT/nets"
  for t in h6 h10; do for s in 1 2 3 4 5 6; do
    python combo/nc_predict.py --weights "$W" --tag $t --subject sub-0$s --eeg "$DATA/tst/sub-0${s}_tst.npy" --out "$OUT/nets"
  done; done
  WF_SCORES=${WF_SCORES:-$OUT/wf} O77_DIR=${O77_DIR:-$OUT/inputs} T2_DIR=${T2_DIR:-$OUT/inputs} \
  NC_DIR="$OUT/nets" NC_FAMILIES=h10:0.075,h6:0.075 python combo/c11.py --out submissions/C13_rebuilt
  # top-1/top-5 agreement with the submitted files (deep ranks may differ on CPU); rows matched per recording
  C13_ALIGN_TO_DATA=1 NC_FAMILIES=h10:0.075,h6:0.075 python combo/c11.py --check submissions/C13_rebuilt
fi
