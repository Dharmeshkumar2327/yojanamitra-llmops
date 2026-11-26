"""Central configuration. Every value can be overridden with an environment variable or .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROMPTS_DIR = ROOT_DIR / "prompts"
SOURCES_FILE = DATA_DIR / "sources.yaml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    # LLM
    groq_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_small_model: str = "llama-3.1-8b-instant"
    judge_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 700
    price_input_per_m: float = 0.59
    price_output_per_m: float = 0.79

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "govt_schemes"

    # Embeddings
    dense_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    sparse_model: str = "Qdrant/bm25"
    rerank_enabled: bool = True
    rerank_model: str = "jinaai/jina-reranker-v2-base-multilingual"

    # Retrieval
    retrieval_mode: str = "hybrid"  # "hybrid" | "dense"  (dense = Week 2 baseline for comparison)
    top_k_prefetch: int = 20
    top_k_final: int = 5
    min_dense_score: float = 0.35

    # Chunking
    chunk_max_words: int = 220
    chunk_overlap_words: int = 40

    # Cache
    redis_url: str = ""
    cache_similarity: float = 0.95
    cache_ttl_seconds: int = 86400

    # Observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    prompt_name: str = "yojanamitra-rag"
    prompt_label: str = "production"
    local_prompt_version: int = 3  # used when Langfuse is not configured

    # Logs
    database_url: str = f"sqlite:///{DATA_DIR / 'requests.db'}"

    # API
    rate_limit_per_min: int = 30

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
