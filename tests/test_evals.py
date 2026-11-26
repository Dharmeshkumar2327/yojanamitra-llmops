import json
from pathlib import Path

from evals.gate import check
from evals.metrics import aggregate, percentile

ROOT = Path(__file__).resolve().parents[1]


def rec(**kw):
    base = dict(
        type="factual",
        expect_refusal=False,
        refused=False,
        faithfulness=1.0,
        relevancy=1.0,
        correctness=1.0,
        n_citations=1,
        expected_scheme="pm_kisan",
        retrieved_schemes=["pm_kisan"],
        latency_ms=1000.0,
        cost_usd=0.001,
    )
    base.update(kw)
    return base


def test_aggregate():
    records = [
        rec(),
        rec(faithfulness=0.2, n_citations=0, retrieved_schemes=["pm_jay"]),
        rec(
            type="out_of_scope",
            expect_refusal=True,
            refused=True,
            faithfulness=None,
            expected_scheme=None,
        ),
        rec(refused=True, faithfulness=None),  # false refusal
    ]
    s = aggregate(records)
    assert s["faithfulness"] == 0.6
    assert s["hallucination_rate"] == 0.5
    assert s["citation_rate"] == 0.5
    assert s["refusal_accuracy"] == 1.0
    assert round(s["false_refusal_rate"], 3) == 0.333
    assert round(s["retrieval_hit_rate"], 3) == 0.667


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([], 0.9) == 0.0


def test_gate():
    base = {"faithfulness": 0.9, "hallucination_rate": 0.05, "correctness": 0.8}
    assert check({"faithfulness": 0.88, "hallucination_rate": 0.06, "correctness": 0.8}, base) == []
    fails = check({"faithfulness": 0.80, "hallucination_rate": 0.10, "correctness": 0.8}, base)
    assert len(fails) == 2


def test_golden_dataset_valid():
    items = [
        json.loads(x)
        for x in (ROOT / "evals/golden_dataset.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids))
    assert sum(i["core"] for i in items) == 30
    for i in items:
        assert {"question", "reference_answer", "type", "expect_refusal", "core"} <= i.keys()
