# neurobyte-web: the NeuroBytes web UI as its own image, separate from the C13 inference image (../docker/).
# Build from the repo root:  docker build -f neurobyte-web/Dockerfile -t ghcr.io/sushruths04/neurobytes-web .
# The challenge data (EEG, dev labels, videos) and captions.json are never baked in; mount them at run time.

# 1. Per-model score matrices (C13, C10, WF) from the repo's own combo/ code and shipped inputs.
FROM python:3.11-slim AS models
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir numpy==2.0.2 scipy==1.14.1 scikit-learn==1.5.2 joblib==1.4.2
WORKDIR /repo
COPY combo combo
COPY src src
COPY results/winfuse results/winfuse
COPY submissions/C13 submissions/C13
COPY neurobyte-web/scripts/export_models.py neurobyte-web/scripts/export_models.py
# Also asserts that C13 sub-05/06 equal submissions/C13 before anything is written.
RUN python neurobyte-web/scripts/export_models.py --team-repo /repo --models C13,C10,WF --out /models

# 2. Frontend bundle (typecheck + Vite build).
FROM node:22-slim AS web
WORKDIR /web
COPY neurobyte-web/package.json neurobyte-web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY neurobyte-web/ ./
RUN npm run build

# 3. Runtime: Node built-ins only (server/serve.ts runs directly with type stripping, Node >= 22.18).
FROM node:22-slim
ENV NODE_ENV=production HOST=0.0.0.0 PORT=8080 \
    NEUROBYTES_DATA_ROOT=/data \
    NEUROBYTES_SUBMISSION_DIR=/app/winfuse \
    NEUROBYTES_MODELS_DIR=/app/models \
    NEUROBYTES_LABEL_LOG=/out/label_access.log
WORKDIR /app/web
COPY --from=web /web/dist dist
COPY --from=web /web/package.json package.json
COPY --from=web /web/server/serve.ts /web/server/dataPlugin.ts /web/server/paths.ts server/
COPY --from=models /models /app/models
COPY results/winfuse /app/winfuse
RUN mkdir -p /data /out && chown node:node /out
USER node
EXPOSE 8080
# No authentication: publish on localhost only, e.g. -p 127.0.0.1:8080:8080
CMD ["node", "--disable-warning=ExperimentalWarning", "server/serve.ts"]
