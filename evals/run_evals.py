"""Run the golden-set evaluation.

    python -m evals.run_evals                       # all items
    python -m evals.run_evals --subset core         # 30 core items (CI)
    python -m evals.run_evals --prompt-version 2    # evaluate a specific prompt version
    python -m evals.run_evals --limit 10 --sleep 2  # quick run, gentle on free rate limits

Writes evals/results/run_<timestamp>.json and evals/results/latest.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from evals.judges import judge_answer
from evals.metrics import aggregate

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "golden_dataset.jsonl"
RESULTS = ROOT / "results"
log = logging.getLogger("evals")


def load_dataset(path: Path, subset: str | None, limit: int | None) -> list[dict]:
    items = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if subset == "core":
        items = [i for i in items if i.get("core")]
    elif subset:
        items = [i for i in items if i.get("type") == subset]
    return items[:limit] if limit else items


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=None, help="core | <type> (e.g. hindi, false_premise)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt-version", type=int, default=None)
    ap.add_argument(
        "--sleep", type=float, default=1.0, help="seconds between items (free-tier rate limits)"
    )
    ap.add_argument("--tag", default="", help="label stored with the run, e.g. 'v3-rerank'")
    args = ap.parse_args()

    from yojanamitra.config import get_settings
    from yojanamitra.generation.factory import build_pipeline
    from yojanamitra.generation.llm import get_llm

    s = get_settings()
    pipeline = build_pipeline(with_cache=False, with_logging=False)
    judge_llm = get_llm()
    items = load_dataset(DATASET, args.subset, args.limit)
    log.info("evaluating %d items (subset=%s)", len(items), args.subset)

    records: list[dict] = []
    for n, item in enumerate(items, start=1):
        resp = pipeline.answer(
            item["question"], use_cache=False, prompt_version=args.prompt_version, log_request=False
        )
        rec = {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "expected_scheme": item.get("expected_scheme"),
            "expect_refusal": item["expect_refusal"],
            "answer": resp.answer,
            "refused": resp.refused,
            "refusal_reason": resp.refusal_reason,
            "retrieved_schemes": resp.retrieved_schemes,
            "n_citations": len(resp.citations),
            "confidence": resp.confidence,
            "latency_ms": resp.latency_ms,
            "cost_usd": resp.cost_usd,
            "prompt_version": resp.prompt_version,
            "faithfulness": None,
            "relevancy": None,
            "correctness": None,
        }
        if item["type"] == "pii":
            digits = re.sub(r"\D", "", item["question"])
            rec["pii_masked"] = "AADHAAR" in resp.pii_found and digits[:12] not in re.sub(
                r"\D", "", resp.answer
            )
        if not item["expect_refusal"] and not resp.refused:
            try:
                rec.update(
                    judge_answer(
                        judge_llm,
                        s.judge_model,
                        item["question"],
                        resp.contexts,
                        resp.answer,
                        item["reference_answer"],
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("judge failed for %s: %s", item["id"], exc)
        records.append(rec)
        log.info(
            "[%d/%d] %s refused=%s faith=%s correct=%s",
            n,
            len(items),
            item["id"],
            resp.refused,
            rec["faithfulness"],
            rec["correctness"],
        )
        time.sleep(args.sleep)

    summary = aggregate(
        [r for r in records if r["refused"] or r["expect_refusal"] or r["faithfulness"] is not None]
    )
    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "subset": args.subset or "all",
        "prompt_version": records[0]["prompt_version"] if records else None,
        "model": s.llm_model,
        "dense_model": s.dense_model,
        "rerank": s.rerank_enabled,
        "summary": summary,
        "records": records,
    }
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (RESULTS / f"run_{stamp}.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (RESULTS / "latest.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n================ EVAL SUMMARY ================")
    for k, v in summary.items():
        if k != "by_type":
            print(f"{k:<26} {v}")
    print("correctness by type:", summary["by_type"])
    print(f"saved -> evals/results/run_{stamp}.json")


if __name__ == "__main__":
    main()
