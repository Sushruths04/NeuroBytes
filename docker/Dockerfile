# NeuroBytes C13: reproducible inference image (CPU) + the neurobyte-web UI.  Build: see docker/README.md
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TMPDIR=/tmp
# Python inference stack + a Node runtime for the web UI's data-API server (serve.ts, Node built-ins only).
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy==2.0.2 scipy==1.14.1 scikit-learn==1.5.2 joblib==1.4.2 \
 && apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && apt-get purge -y --auto-remove curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY context/repo /app/repo
COPY context/weights /app/weights
COPY context/winfuse_scores /app/winfuse_scores
# neurobyte-web: prebuilt frontend bundle, data-API server, and per-model matrices (staged by build.sh).
# These are optional: the directories may be empty if the web bundle was not staged (SKIP_WEB=1).
COPY context/web /app/web
COPY context/neurobytes_models /app/neurobytes_models
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh && mkdir -p /out /data
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["help"]
