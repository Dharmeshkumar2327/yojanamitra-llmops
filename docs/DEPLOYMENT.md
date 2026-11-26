# Deployment guide (free tier)

| Layer | Service | Notes |
|---|---|---|
| LLM | Groq API | Free tier is rate-limited; over the limit you get HTTP 429 (the client retries with backoff) |
| Vector DB | Qdrant Cloud free cluster | 1 GB RAM / 4 GB disk - plenty for ~20 schemes |
| Tracing + prompts | Langfuse Cloud Hobby | 50k units/month. Teachers/students can request the Langfuse Fast Grant |
| Request logs | Neon or Supabase free Postgres | Any `postgresql://` URL works as `DATABASE_URL` |
| Cache (optional) | Upstash free Redis | Empty `REDIS_URL` = in-memory cache |
| API | Render free web service | Sleeps after 15 min idle; first request ~1 min. Reranker OFF (512 MB RAM) |
| UI | Hugging Face Space (Docker) | Public demo link |
| CI/CD | GitHub Actions | Free for public repos |

> Free-tier limits change - check each provider's pricing page before relying on them.

---

## 1. Qdrant Cloud: index your data once

1. cloud.qdrant.io → **Create cluster → Free** → copy the cluster URL and create an API key.
2. In `.env`: `QDRANT_URL=https://xxxx.cloud.qdrant.io` and `QDRANT_API_KEY=...`
3. `python -m yojanamitra.ingestion.pipeline --index --recreate --no-fetch`

## 2. Langfuse

1. cloud.langfuse.com → new project → **Settings → API keys** → copy both.
2. `python scripts/push_prompts_to_langfuse.py`

## 3. Postgres for request logs (so the dashboard survives restarts)

Render's free filesystem is wiped on every restart, so SQLite there would lose data.
Create a free Postgres on Neon (neon.tech) or Supabase → copy the connection string →
use it as `DATABASE_URL` (format `postgresql://user:pass@host/db`). Tables are created automatically.

## 4. API on Render

1. render.com → **New → Blueprint** → connect your GitHub repo → it reads `render.yaml`.
2. Enter the secret env vars when asked (`GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`,
   `LANGFUSE_*`, `DATABASE_URL`, optional `REDIS_URL`).
3. After the first deploy: open `https://<name>.onrender.com/health` → `"qdrant": true`.
4. Service → **Settings → Deploy Hook** → copy the URL → GitHub secret `RENDER_DEPLOY_HOOK_URL`.
   (`autoDeploy: false` in render.yaml means deploys only happen after CI passes.)

## 5. UI on Hugging Face Spaces

1. huggingface.co → **New Space** → SDK **Docker** → blank → name `yojanamitra`.
2. Space **Settings → Variables and secrets**, choose ONE mode:
   - **UI only:** `API_URL=https://<name>.onrender.com`
   - **All-in-one (no cold starts, recommended):** `YOJANAMITRA_LOCAL=1`, `GROQ_API_KEY`,
     `QDRANT_URL`, `QDRANT_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `DATABASE_URL`
3. huggingface.co → **Settings → Access Tokens** → new token with *write* → GitHub secrets
   `HF_TOKEN` and `HF_SPACE` (= `your-username/yojanamitra`).
4. Push to `main` → CI passes → `deploy.yml` uploads the Space → it builds in a few minutes.

## 6. GitHub secrets checklist

| Secret | Used by |
|---|---|
| `GROQ_API_KEY` | eval gate, weekly refresh |
| `QDRANT_URL`, `QDRANT_API_KEY` | eval gate, weekly refresh |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | eval gate (prompt from registry) |
| `RENDER_DEPLOY_HOOK_URL` | deploy |
| `HF_TOKEN`, `HF_SPACE` | deploy |

Never commit `.env`. `.gitignore` already excludes it and pre-commit's `detect-private-key` hook helps.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `429 Too Many Requests` from Groq during evals | `--sleep 3`, use `--subset core`, or run evals later |
| Render out of memory | keep `RERANK_ENABLED=false`; the default MiniLM dense model is the small one |
| Every answer is "could not find" | collection empty or wrong `QDRANT_URL`; check `/health` and Qdrant dashboard; lower `MIN_DENSE_SCORE` slightly (0.30) |
| Hindi answers weak | add Hindi PDFs (many ministries publish Hindi guidelines) as `data/raw/<id>_hi.pdf` |
| Website gives little text | it's JavaScript-rendered - save the page as PDF into `data/raw/` |
| CI lint fails | run `ruff check --fix . && ruff format .` locally, then commit |
