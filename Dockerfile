# syntax=docker/dockerfile:1
# Free-stack image: CPU torch, open-weight models baked in, non-root, no data baked in.
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
COPY requirements-core.txt requirements-ml.txt requirements-ui.txt ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-core.txt -r requirements-ml.txt -r requirements-ui.txt
ENV HF_HOME=/models
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
 && python -c "import nltk; nltk.download('punkt', download_dir='/opt/nltk'); nltk.download('punkt_tab', download_dir='/opt/nltk')"

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/models NLTK_DATA=/opt/nltk EARNINGS_DATA_DIR=/data PORT=7860
RUN useradd -m -u 1000 app && mkdir -p /data && chown app /data
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /models /models
COPY --from=builder /opt/nltk /opt/nltk
WORKDIR /app
COPY --chown=app src ./src
COPY --chown=app app ./app
COPY --chown=app scripts ./scripts
COPY --chown=app eval ./eval
USER app
VOLUME ["/data"]
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"API_PORT\", os.environ[\"PORT\"])}/health')" || exit 1
# MODE=api (default) | ui | all (API on :8000 + UI on :$PORT; use "all" for a single-port host like HF Spaces)
CMD ["bash", "scripts/start.sh"]
