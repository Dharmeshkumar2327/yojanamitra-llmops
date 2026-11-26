# Hugging Face Space (Docker SDK). Two ways to run it:
#  - UI only  : set Space secret API_URL=https://<your-render-app>.onrender.com
#  - All-in-one (recommended on free tier, no cold starts): set YOJANAMITRA_LOCAL=1
#    plus GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY, LANGFUSE_* as Space secrets.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed \
    HOME=/home/user

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements.txt streamlit pandas requests

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
COPY --chown=user prompts ./prompts
COPY --chown=user ui ./ui
COPY --chown=user data/sources.yaml ./data/sources.yaml
COPY --chown=user evals ./evals
RUN pip install --no-deps -e . && mkdir -p /app/data && chown -R user /app

USER user
EXPOSE 7860
CMD ["streamlit", "run", "ui/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
