"""Aggregate per-item eval records into run-level metrics (pure python, unit-tested)."""

from __future__ import annotations

import statistics


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def _mean(xs: list[float]) -> float:
    return round(statistics.fmean(xs), 4) if xs else 0.0


def aggregate(records: list[dict], hallucination_threshold: float = 0.5) -> dict:
    answerable = [r for r in records if not r["expect_refusal"]]
    refusal_items = [r for r in records if r["expect_refusal"]]
    answered = [r for r in answerable if not r["refused"] and r.get("faithfulness") is not None]
    with_scheme = [r for r in answerable if r.get("expected_scheme")]
    pii_items = [r for r in records if r["type"] == "pii"]
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]

    return {
        "n_items": len(records),
        "faithfulness": _mean([r["faithfulness"] for r in answered]),
        "answer_relevancy": _mean([r["relevancy"] for r in answered]),
        "correctness": _mean([r["correctness"] for r in answered]),
        "hallucination_rate": _mean(
            [1.0 if r["faithfulness"] < hallucination_threshold else 0.0 for r in answered]
        ),
        "citation_rate": _mean([1.0 if r["n_citations"] > 0 else 0.0 for r in answered]),
        "retrieval_hit_rate": _mean(
            [1.0 if r["expected_scheme"] in r["retrieved_schemes"] else 0.0 for r in with_scheme]
        ),
        "refusal_accuracy": _mean([1.0 if r["refused"] else 0.0 for r in refusal_items]),
        "false_refusal_rate": _mean([1.0 if r["refused"] else 0.0 for r in answerable]),
        "pii_mask_rate": _mean([1.0 if r.get("pii_masked") else 0.0 for r in pii_items]),
        "latency_p50_ms": round(percentile(latencies, 0.5), 1),
        "latency_p95_ms": round(percentile(latencies, 0.95), 1),
        "avg_cost_usd": round(statistics.fmean([r["cost_usd"] for r in records]), 6)
        if records
        else 0.0,
        "cost_per_1k_queries_usd": round(
            1000 * statistics.fmean([r["cost_usd"] for r in records]), 4
        )
        if records
        else 0.0,
        "by_type": {
            t: _mean([r["correctness"] for r in answered if r["type"] == t])
            for t in sorted({r["type"] for r in answered})
        },
    }
