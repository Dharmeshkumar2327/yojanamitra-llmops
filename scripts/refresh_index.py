"""Incremental re-indexing ("data drift" handling for a RAG system).

For every source document: compute the content hash, compare with what is stored in
Qdrant, and re-index only changed or new documents. Reports what changed.

    python scripts/refresh_index.py
    python scripts/refresh_index.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging

log = logging.getLogger("refresh")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    from yojanamitra.config import RAW_DIR, get_settings
    from yojanamitra.ingestion.loaders import load_scheme_documents
    from yojanamitra.ingestion.pipeline import build_chunks
    from yojanamitra.ingestion.registry import load_schemes
    from yojanamitra.retrieval.vector_store import get_vector_store

    s = get_settings()
    store = get_vector_store()
    store.ensure_collection()
    indexed = store.source_hashes()
    report = {"unchanged": [], "updated": [], "new": [], "failed": []}

    for scheme in load_schemes():
        for doc in load_scheme_documents(scheme, RAW_DIR, allow_fetch=True):
            old = indexed.get(doc.source_id)
            if old == doc.content_hash:
                report["unchanged"].append(doc.source_id)
                continue
            status = "new" if old is None else "updated"
            chunks = build_chunks(doc, s.chunk_max_words, s.chunk_overlap_words)
            if not chunks:
                report["failed"].append(doc.source_id)
                continue
            if not args.dry_run:
                store.delete_source(doc.source_id)
                store.upsert_chunks([c.to_dict() for c in chunks])
            report[status].append(doc.source_id)
            log.info("%s %s (%d chunks)", status.upper(), doc.source_id, len(chunks))

    summary = {k: len(v) for k, v in report.items()}
    log.info("refresh summary: %s", summary)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, **report}, f, indent=2)


if __name__ == "__main__":
    main()
