#!/bin/bash
# NeuroBytes C13 container. Mount the official challenge folder at /data (needs tst/ ; tst_stim/ enables evaluation)
# and an output folder at /out.
set -e
R=/app/repo; W=/app/weights
export WINFUSE_DATA_ROOT=/data WF_SCORES=/app/winfuse_scores NC_FAMILIES=h10:0.075,h6:0.075 C13_ALIGN_TO_DATA=1
case "$1" in
  replay)
    cp $R/submissions/C13/*.npy /out/ && echo "C13 ranked lists (frozen) written to /out" ;;
  predict)
    # networks recomputed from the mounted raw test EEG with the trained weights; ridge/T2 component scores shipped
    mkdir -p /tmp/nets
    for t in h6 h10; do for s in ${SUBJECTS:-1 2 3 4 5 6}; do
      [ -f /data/tst/sub-0${s}_tst.npy ] && python $R/combo/nc_predict.py --weights $W --tag $t --subject sub-0$s \
          --eeg /data/tst/sub-0${s}_tst.npy --out /tmp/nets
    done; done
    NC_DIR=/tmp/nets python $R/combo/c11.py --out /out
    NC_DIR=/tmp/nets python $R/combo/c11.py --check /out
    ;;
  evaluate)
    mkdir -p /tmp/nets
    for t in h6 h10; do for s in 1 2 3 4; do python $R/combo/nc_predict.py --weights $W --tag $t --subject sub-0$s \
        --eeg /data/tst/sub-0${s}_tst.npy --out /tmp/nets; done; done
    NC_DIR=/tmp/nets python $R/combo/c11.py --dev ;;
  web)
    # Serve the neurobyte-web UI + read-only data API (node built-ins only; no npm/vite at runtime).
    # Ranking uses the repo's own artefacts baked into the image (winfuse scores + model matrices);
    # the optional /data:ro mount adds the raw EEG, dev labels and stimulus videos.
    [ -f /app/web/server/serve.ts ] || { echo "web bundle not in this image (built with SKIP_WEB=1 or without node). Rebuild with docker/build.sh."; exit 1; }
    export HOST=0.0.0.0 PORT="${PORT:-8080}" \
      NEUROBYTES_DATA_ROOT=/data \
      NEUROBYTES_SUBMISSION_DIR="$R/results/winfuse" \
      NEUROBYTES_MODELS_DIR=/app/neurobytes_models \
      NEUROBYTES_LABEL_LOG=/out/label_access.log
    echo "neurobyte-web on http://127.0.0.1:${PORT} (publish with -p 127.0.0.1:${PORT}:${PORT})"
    exec node --disable-warning=ExperimentalWarning /app/web/server/serve.ts ;;
  *)
    echo "usage: docker run --rm -v <challenge_data>:/data:ro -v <out>:/out ghcr.io/sushruths04/neurobytes {replay|predict|evaluate|web}"
    echo "  replay   : write the frozen C13 ranked lists"
    echo "  predict  : recompute the networks from /data/tst with the trained weights and assemble C13 (sub-05/06 lists)"
    echo "  evaluate : dev-test top-1/top-5 on sub-01..04 (needs /data/tst and the shipped dev labels)"
    echo "  web      : serve the neurobyte-web UI + data API on PORT (default 8080); publish with -p 127.0.0.1:8080:8080." ;;
esac
