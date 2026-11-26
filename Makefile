.PHONY: install up down ingest index eval eval-core gate prompts serve ui test lint load refresh

install:
	pip install -r requirements-dev.txt && pip install -e . && pre-commit install

up:            ## start qdrant + redis locally
	docker compose up -d qdrant redis

down:
	docker compose down

ingest:        ## fetch/parse/chunk docs -> data/processed/chunks.jsonl
	python -m yojanamitra.ingestion.pipeline

index:         ## ingest + push to Qdrant
	python -m yojanamitra.ingestion.pipeline --index --recreate

prompts:       ## push local prompt versions to Langfuse
	python scripts/push_prompts_to_langfuse.py

eval:          ## full golden-set evaluation
	python -m evals.run_evals

eval-core:     ## 30 core questions (used in CI)
	python -m evals.run_evals --subset core

gate:          ## fail if quality regressed vs baseline
	python -m evals.gate --results evals/results/latest.json

serve:
	uvicorn yojanamitra.api.main:app --reload --port 8000

ui:
	streamlit run ui/app.py

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

load:
	locust -f loadtest/locustfile.py --host http://localhost:8000

refresh:
	python scripts/refresh_index.py
