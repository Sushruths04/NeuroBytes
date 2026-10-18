# NeuroBytes web — EEG → video retrieval frontend + read-only data API.
#
# Data handling follows the repo convention (see ../README.md "Data" and
# ../docker/README.md): the official CHALLENGE DATA is never baked into the image.
# It is mounted read-only at /data at run time, exactly like the C13 inference image:
#
#   # from the neurobytes repo root, after running export_models.py once:
#   docker build -t neurobyte-web -f neurobyte-web/Dockerfile .
#   D=/path/to/eeg-mind-reading-challenge
#   docker run --rm -p 127.0.0.1:8080:8080 -v "$D:/data:ro" -v neurobyte-web-logs:/logs neurobyte-web
#   # open http://127.0.0.1:8080
#
# Only the repo's OWN shipped artefacts (results/winfuse fused scores and the exported
# results/neurobytes_models matrices) are copied into the image — the same artefacts the
# repo's build.sh packages. The raw EEG (tst/), dev labels (tst_stim/) and stimulus
# videos (stimuli_test/) come from the /data:ro mount; without it those endpoints 404.

ARG NODE_IMAGE=node:24.11.1-alpine3.22

# ---- build: install, production build (frontend bundle) ----
FROM ${NODE_IMAGE} AS build
WORKDIR /app
COPY neurobyte-web/package.json neurobyte-web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY neurobyte-web/ ./
RUN npm run build

# ---- runtime: dist/ + server/ + the repo's own artefacts; challenge data is mounted at /data ----
FROM ${NODE_IMAGE}
ENV NODE_ENV=production \
    HOST=0.0.0.0 \
    PORT=8080 \
    NEUROBYTES_DATA_ROOT=/data \
    NEUROBYTES_SUBMISSION_DIR=/app/results/winfuse \
    NEUROBYTES_MODELS_DIR=/app/results/neurobytes_models \
    NEUROBYTES_LABEL_LOG=/logs/label_access.log
WORKDIR /app
COPY --from=build /app/dist ./dist
COPY --from=build /app/server/serve.ts /app/server/dataPlugin.ts /app/server/paths.ts ./server/
COPY --from=build /app/package.json ./
# The repo's OWN shipped artefacts (not challenge data). Build context is the repo root.
COPY results/winfuse ./results/winfuse
COPY results/neurobytes_models ./results/neurobytes_models
RUN mkdir -p /logs /data && chown node:node /logs
USER node
EXPOSE 8080
# Challenge data mounts here read-only (-v "$D:/data:ro"); may be empty if unused.
VOLUME ["/logs"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD wget -q -O /dev/null "http://127.0.0.1:${PORT}/" || exit 1
CMD ["node", "--disable-warning=ExperimentalWarning", "server/serve.ts"]
