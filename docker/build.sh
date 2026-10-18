#!/bin/bash
# Build the NeuroBytes C13 image. Run from anywhere:  bash docker/build.sh
# Needs: docker, and either gh (to download the v1.0 release) or WEIGHTS_TAR=/path/to/c13o_weights_20261018.tar.gz
# Works from a git clone (packages the committed HEAD) or from a "Download ZIP" copy (packages the folder as is).
# CONTEXT_ONLY=1 prepares docker/context without building (used by .github/workflows/docker.yml).
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd); CTX=$REPO/docker/context
rm -rf "$CTX" && mkdir -p "$CTX/repo" "$CTX/winfuse_scores"
if git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  [[ -n $(git -C "$REPO" status --porcelain -- . ':!docker/context') ]] && echo "WARNING: uncommitted changes are NOT included (git archive HEAD)"
  git -C "$REPO" archive HEAD | tar -x -C "$CTX/repo"
else
  tar -C "$REPO" --exclude=./docker/context --exclude=./.venv --exclude=./.git -cf - . | tar -x -C "$CTX/repo"
fi
if [[ -n "${WEIGHTS_TAR:-}" ]]; then cp "$WEIGHTS_TAR" "$CTX/"
else gh release download v1.0 -R Sushruths04/neurobytes -p 'c13o_weights_20261018.tar.gz' -D "$CTX"; fi
echo "c81efde41c2ee259160e48c5242d215b1673cc3a6233455ae422937679a3ab3f  $CTX/c13o_weights_20261018.tar.gz" | sha256sum -c -
tar -xzf "$CTX/c13o_weights_20261018.tar.gz" -C "$CTX" && mv "$CTX/c13o_weights_20261018" "$CTX/weights" && rm "$CTX/c13o_weights_20261018.tar.gz"
cp "$CTX"/repo/results/winfuse/scores/*.npy "$CTX/winfuse_scores/"      # aligned winfuse fused scores shipped in the repo
[[ ${CONTEXT_ONLY:-0} == 1 ]] && { echo "context ready: $CTX"; exit 0; }
docker build -t ghcr.io/sushruths04/neurobytes:latest "$REPO/docker"
echo "built ghcr.io/sushruths04/neurobytes:latest"
