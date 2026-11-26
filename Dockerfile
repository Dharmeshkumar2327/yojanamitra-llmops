FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY prompts ./prompts
COPY data/sources.yaml ./data/sources.yaml
RUN pip install --no-deps -e .

# Bake embedding models into the image -> no download on cold start
ARG DENSE_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ARG RERANK=false
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
TextEmbedding('${DENSE_MODEL}'); SparseTextEmbedding('Qdrant/bm25')" && \
    if [ "$RERANK" = "true" ]; then python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextCrossEncoder('jinaai/jina-reranker-v2-base-multilingual')"; fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fs http://localhost:${PORT:-8000}/health || exit 1
CMD ["sh", "-c", "uvicorn yojanamitra.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
