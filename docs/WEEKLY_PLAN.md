# YojanaMitra — 6-Week Build & GitHub Commit Plan

All code for all six weeks is already in this repo. Use this guide to **build and understand it
week by week** and to **commit in stages**, so your GitHub history shows real progress (recruiters
look at it). Each week lists: goal → files → commands → what to check → the git commits.

> Time: ~10–12 hrs/week. Python 3.10+ and Docker Desktop are needed.

---

## Week 0 — One-time setup (1–2 hrs)

1. Create an empty **public** GitHub repo named `yojanamitra-llmops` (no README, no .gitignore).
2. Create free accounts and copy the keys into `.env` (from `.env.example`):
   - **Groq** → console.groq.com → API Keys → `GROQ_API_KEY`
   - **Qdrant Cloud** → cloud.qdrant.io → create **Free** cluster → URL + API key (needed in Week 5–6; use local Docker before that)
   - **Langfuse** → cloud.langfuse.com → new project → public + secret keys (needed from Week 4)
3. Install:
   ```bash
   python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
   pip install -r requirements-dev.txt && pip install -e .
   pre-commit install
   ```

### Staging trick (to commit week by week)
Unzip the project into a separate folder (e.g. `yojanamitra-full/`). Start your repo folder
**empty** and copy files in from the full folder each week, as listed below. Then commit.

```bash
mkdir yojanamitra-llmops && cd yojanamitra-llmops
git init -b main
git remote add origin https://github.com/Dharmeshkumar2327/yojanamitra-llmops.git
```

---

## Week 1 — Data pipeline & repo setup

**Goal:** turn 15–20 scheme documents into clean, section-labelled chunks.

**Copy in:**
```
.gitignore  .pre-commit-config.yaml  .env.example  LICENSE  Makefile  pyproject.toml
requirements.txt  requirements-dev.txt  README.md
data/sources.yaml  data/raw/.gitkeep  data/processed/.gitkeep
src/yojanamitra/__init__.py  src/yojanamitra/config.py
src/yojanamitra/ingestion/   (all files)
tests/conftest.py  tests/test_ingestion.py
```

**Do:**
1. For each scheme in `data/sources.yaml`, open it on **myscheme.gov.in** → Print → *Save as PDF* →
   save as `data/raw/<id>.pdf` (e.g. `data/raw/pm_kisan.pdf`). Add official guideline PDFs as
   `data/raw/<id>_guidelines.pdf`. Aim for 15–20 schemes.
2. Run:
   ```bash
   python -m yojanamitra.ingestion.pipeline --no-fetch
   pytest tests/test_ingestion.py -q
   ```
3. Open `data/processed/chunks.jsonl` and **read 20 chunks**. Are sections right? Is text clean?
   Tune `SECTION_KEYWORDS` in `chunker.py` if a scheme's headings are missed.

**Learn:** why section-aware chunking beats fixed-size chunks for eligibility questions; why
`unicodedata.normalize("NFC")` matters for Hindi.

**Commits:**
```bash
git add .gitignore .pre-commit-config.yaml .env.example LICENSE Makefile pyproject.toml requirements*.txt README.md
git commit -m "chore: project setup, tooling and config"
git add data/ src/ tests/
git commit -m "feat(ingestion): loaders, cleaner and section-aware chunker for 20 scheme docs"
git push -u origin main
```

---

## Week 2 — Baseline RAG

**Goal:** first working question → answer with citations.

**Copy in:**
```
src/yojanamitra/retrieval/      (embeddings, types, vector_store, reranker, router)
src/yojanamitra/generation/     (llm, prompts, rag, factory)
src/yojanamitra/guardrails/     (needed by rag.py - you'll study it in Week 4)
src/yojanamitra/cache/  src/yojanamitra/observability/  src/yojanamitra/monitoring/
prompts/rag_v1.yaml  prompts/rag_v2.yaml
docker-compose.yml
```

**Do:**
```bash
docker compose up -d qdrant redis
python -m yojanamitra.ingestion.pipeline --index --recreate --no-fetch
python - <<'EOF'
from yojanamitra.generation.factory import build_pipeline
p = build_pipeline()
for q in ["Who is eligible for PM-KISAN?", "PMSBY premium kitna hai?", "सुकन्या समृद्धि में न्यूनतम जमा?"]:
    r = p.answer(q); print(q, "\n->", r.answer, "\n", [c.id for c in r.citations], "\n")
EOF
```
Open http://localhost:6333/dashboard to see your collection.

**Experiment (for the README results table):** set `RETRIEVAL_MODE=dense`, `RERANK_ENABLED=false`,
`LOCAL_PROMPT_VERSION=1` → this is your **v1 baseline**. You will measure it in Week 3.

**Commits:**
```bash
git add src/yojanamitra/retrieval src/yojanamitra/generation prompts/rag_v1.yaml docker-compose.yml
git commit -m "feat(rag): hybrid dense+BM25 retrieval with RRF, scheme router, Groq generation"
git add prompts/rag_v2.yaml src/yojanamitra/guardrails src/yojanamitra/cache src/yojanamitra/observability src/yojanamitra/monitoring
git commit -m "feat(rag): grounded prompt v2 with citations and refusal"
git push
```

---

## Week 3 — Evaluation system (the heart of LLMOps)

**Goal:** a number for "how good is the bot", and the ability to compare versions.

**Copy in:**
```
evals/   (all files: golden_dataset.jsonl, run_evals.py, judges.py, metrics.py, gate.py,
          baseline.json, generate_synthetic.py, __init__.py, results/.gitkeep)
tests/test_evals.py
```

**Do:**
1. **Verify every reference answer** in `evals/golden_dataset.jsonl` against your PDFs
   (amounts change - the documents are the truth). Fix any that differ.
2. Grow the set to 100–150 items:
   ```bash
   python -m evals.generate_synthetic --n 80
   # review evals/synthetic_candidates.jsonl by hand, keep the good ones, append to golden_dataset.jsonl
   ```
3. Run the three versions and save each run:
   ```bash
   RETRIEVAL_MODE=dense RERANK_ENABLED=false python -m evals.run_evals --prompt-version 1 --tag v1-baseline
   RERANK_ENABLED=false python -m evals.run_evals --prompt-version 2 --tag v2-hybrid
   python -m evals.run_evals --prompt-version 3 --tag v3-full
   python -m evals.gate --results evals/results/latest.json --update-baseline   # lock in v3 as baseline
   ```
4. Put the three rows into the README results table.

**Learn:** faithfulness vs correctness (an answer can be faithful to a wrong chunk); why
false-refusal rate matters as much as hallucination rate.

**Commits:**
```bash
git add evals/ tests/test_evals.py
git commit -m "feat(evals): golden dataset, LLM-as-judge metrics, quality gate"
git add evals/baseline.json README.md
git commit -m "docs: baseline vs hybrid vs rerank eval results"
git push
```

---

## Week 4 — Prompt versioning, tracing, guardrails, cache

**Goal:** see inside every request; protect users; cut cost.

**Copy in:**
```
prompts/rag_v3.yaml
scripts/push_prompts_to_langfuse.py
tests/test_guardrails.py  tests/test_rag_pipeline.py
```
(the guardrails / cache / tracing code is already in from Week 2 - this week you **turn it on and study it**)

**Do:**
1. Add Langfuse keys to `.env`, then:
   ```bash
   python scripts/push_prompts_to_langfuse.py           # v1, v2, v3 -> Langfuse, v3 = production
   ```
2. Ask 20 questions → open Langfuse → **Traces**: see guardrails → retrieval → rerank → answer,
   with tokens, latency and cost. Screenshot one trace for the README.
3. Try a rollback: `python scripts/push_prompts_to_langfuse.py --production 2` → the app now uses v2
   without a redeploy. Switch back to 3.
4. Test guardrails: Aadhaar `2345 6789 0124` gets masked; "ignore previous instructions…" is
   blocked; "who won IPL" is refused.
5. Ask the same question twice → second answer shows `cache_hit=True`, cost 0.

**Commits:**
```bash
git add prompts/rag_v3.yaml
git commit -m "feat(prompts): v3 with Hindi/English matching, false-premise correction, eligibility format"
git add scripts/push_prompts_to_langfuse.py
git commit -m "feat(llmops): Langfuse prompt registry, request tracing and cost tracking"
git add tests/test_guardrails.py tests/test_rag_pipeline.py
git commit -m "test: guardrails (Verhoeff Aadhaar, PII, injection, router) and end-to-end pipeline with fakes"
git push
```

---

## Week 5 — API, UI and CI/CD

**Goal:** a product people can use, and a pipeline that blocks bad changes.

**Copy in:**
```
src/yojanamitra/api/   ui/   tests/test_api.py   tests/test_cache_and_limits.py
Dockerfile   hf_space/   .github/workflows/ci.yml
```

**Do:**
```bash
make serve        # http://localhost:8000/docs  -> try /ask, /feedback, /stats
make ui           # http://localhost:8501
docker compose up --build   # whole stack in containers
```
Then GitHub → repo **Settings → Secrets and variables → Actions → New secret**:
`GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`.
(CI uses Qdrant Cloud, so index into Cloud once: set `QDRANT_URL`/`QDRANT_API_KEY` in `.env` and run `make index`.)

**Prove the gate works (great for the README/LinkedIn):**
```bash
git checkout -b break-prompt
# edit prompts/rag_v3.yaml: delete rule 1 ("Use ONLY the numbered context passages")
git commit -am "test: weaken grounding rule" && git push -u origin break-prompt
# open a PR -> watch the eval-gate job FAIL on faithfulness/hallucination -> screenshot -> close PR
```

**Commits:**
```bash
git add src/yojanamitra/api tests/test_api.py tests/test_cache_and_limits.py
git commit -m "feat(api): FastAPI /ask /feedback /stats /health with rate limiting"
git add ui/
git commit -m "feat(ui): Streamlit chat with citations and feedback + monitoring dashboard"
git add Dockerfile hf_space/ .github/workflows/ci.yml
git commit -m "ci: lint, tests and eval-gated quality check on every PR"
git push
```

---

## Week 6 — Deploy live, load test, monitor, polish

**Goal:** a public link, numbers, and a README that sells it.

**Copy in:**
```
render.yaml  .github/workflows/deploy.yml  .github/workflows/refresh-index.yml
scripts/refresh_index.py  loadtest/  docs/
```

**Do:**
1. Deploy following **docs/DEPLOYMENT.md** (Render API + Hugging Face Space).
2. Load test locally (short runs - free Groq limits):
   ```bash
   RATE_LIMIT_PER_MIN=1000 make serve
   locust -f loadtest/locustfile.py --host http://localhost:8000 --users 10 --spawn-rate 2 --run-time 2m --headless
   ```
   Record p50/p95 latency and cache hit rate.
3. Run the weekly refresh once by hand: GitHub → Actions → *Weekly index refresh* → Run workflow.
4. Record a 30-second demo GIF (ScreenToGif / Kap): English question → Hindi question → PII masked →
   off-topic refused → monitoring page. Save as `docs/demo.gif`.
5. Finish the README: live links, results table, Langfuse trace screenshot, failed-gate screenshot,
   cost per 1k queries.

**Commits:**
```bash
git add render.yaml .github/workflows/deploy.yml
git commit -m "cd: auto-deploy to Render and Hugging Face Spaces after CI passes"
git add .github/workflows/refresh-index.yml scripts/refresh_index.py
git commit -m "feat(ops): weekly incremental re-indexing with post-refresh eval gate"
git add loadtest/ docs/
git commit -m "perf: locust load test; docs: deployment guide and weekly plan"
git add README.md docs/demo.gif
git commit -m "docs: v1.0 README with live demo, architecture, eval results"
git tag -a v1.0.0 -m "YojanaMitra v1.0" && git push --follow-tags
```
Then GitHub → **Releases → Draft new release** from tag `v1.0.0`.

---

## LinkedIn post skeleton (after Week 6)

> Built **YojanaMitra** 🇮🇳 — an LLMOps RAG assistant that answers questions on 20 Indian Government
> schemes in English & Hindi, only from official documents, with citations.
> What makes it production-grade: eval-gated CI (a PR that hurts faithfulness is blocked), prompt
> versioning with instant rollback, full tracing + cost per query, PII masking for Aadhaar/PAN, and a
> weekly re-indexing job. Faithfulness X% · hallucination Y% · ₹Z per 1,000 queries.
> Live demo + code 👇
