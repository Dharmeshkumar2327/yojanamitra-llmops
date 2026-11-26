"""Ingestion CLI.

    python -m yojanamitra.ingestion.pipeline                 # parse + chunk -> data/processed/chunks.jsonl
    python -m yojanamitra.ingestion.pipeline --index         # ... and upsert to Qdrant
    python -m yojanamitra.ingestion.pipeline --index --recreate
    python -m yojanamitra.ingestion.pipeline --no-fetch      # only use files in data/raw
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from yojanamitra.ingestion.chunker import Chunk, chunk_document
from yojanamitra.ingestion.cleaner import clean_pages
from yojanamitra.ingestion.loaders import RawDocument, load_scheme_documents
from yojanamitra.ingestion.registry import load_schemes

log = logging.getLogger("ingestion")


def build_chunks(doc: RawDocument, max_words: int, overlap: int) -> list[Chunk]:
    text = clean_pages(doc.pages)
    return chunk_document(
        text,
        scheme_id=doc.scheme_id,
        scheme_name=doc.scheme_name,
        source_id=doc.source_id,
        source_url=doc.source_url,
        ministry=doc.ministry,
        doc_hash=doc.content_hash,
        max_words=max_words,
        overlap=overlap,
    )


def run(index: bool, recreate: bool, allow_fetch: bool) -> list[Chunk]:
    from yojanamitra.config import PROCESSED_DIR, RAW_DIR, get_settings

    s = get_settings()
    all_chunks: list[Chunk] = []
    missing: list[str] = []
    for scheme in load_schemes():
        docs = load_scheme_documents(scheme, RAW_DIR, allow_fetch=allow_fetch)
        if not docs:
            missing.append(scheme.id)
            continue
        for doc in docs:
            all_chunks.extend(build_chunks(doc, s.chunk_max_words, s.chunk_overlap_words))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(PROCESSED_DIR) / "chunks.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    by_scheme = Counter(c.scheme_id for c in all_chunks)
    by_section = Counter(c.section for c in all_chunks)
    log.info("wrote %d chunks -> %s", len(all_chunks), out)
    log.info("chunks per scheme: %s", dict(by_scheme))
    log.info("chunks per section: %s", dict(by_section))
    if missing:
        log.warning("NO documents for: %s  (save PDFs into data/raw/<id>.pdf)", ", ".join(missing))

    if index and all_chunks:
        from yojanamitra.retrieval.vector_store import get_vector_store

        store = get_vector_store()
        store.ensure_collection(recreate=recreate)
        n = store.upsert_chunks([c.to_dict() for c in all_chunks])
        log.info("upserted %d points into Qdrant collection '%s' (total=%d)", n, store.collection, store.count())
    return all_chunks


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="YojanaMitra ingestion")
    p.add_argument("--index", action="store_true", help="upsert chunks into Qdrant")
    p.add_argument("--recreate", action="store_true", help="drop and recreate the collection")
    p.add_argument("--no-fetch", action="store_true", help="do not fetch URLs, only data/raw files")
    args = p.parse_args()
    run(index=args.index, recreate=args.recreate, allow_fetch=not args.no_fetch)


if __name__ == "__main__":
    main()
