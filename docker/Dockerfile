# NeuroBytes C13: reproducible inference image (CPU).  Build: see docker/README.md
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TMPDIR=/tmp
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy==2.0.2 scipy==1.14.1 scikit-learn==1.5.2 joblib==1.4.2
WORKDIR /app
COPY context/repo /app/repo
COPY context/weights /app/weights
COPY context/winfuse_scores /app/winfuse_scores
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh && mkdir -p /out /data
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["help"]
