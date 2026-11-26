"""LLM-as-judge metrics (RAGAS-style, no extra dependency).

One judge call per answered item returns:
    faithfulness  - share of answer claims supported by the retrieved context (0..1)
    relevancy     - does the answer address the question (0..1)
    correctness   - agreement with the reference answer (0..1)
"""

from __future__ import annotations

JUDGE_PROMPT = """You are a strict evaluator of a RAG system for Indian Government schemes.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

SYSTEM ANSWER:
{answer}

REFERENCE ANSWER (ground truth written by a human):
{reference}

Score each metric from 0.0 to 1.0:
- faithfulness: split the SYSTEM ANSWER into factual claims. What fraction is directly supported
  by the RETRIEVED CONTEXT? (Claims only supported by outside knowledge are NOT supported.)
- relevancy: how well does the SYSTEM ANSWER address the QUESTION?
- correctness: how well does the SYSTEM ANSWER agree with the REFERENCE ANSWER on the key facts
  (amounts, ages, limits, yes/no)? Extra correct detail is fine; contradictions score low.

Return JSON only:
{{"faithfulness": float, "relevancy": float, "correctness": float, "unsupported_claims": [string]}}"""


def _clip(x) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def judge_answer(
    llm, model: str, question: str, contexts: list[str], answer: str, reference: str
) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        context="\n---\n".join(contexts)[:9000] or "(none)",
        answer=answer,
        reference=reference,
    )
    res = llm.chat(
        [{"role": "user", "content": prompt}],
        model=model,
        temperature=0,
        max_tokens=400,
        json_mode=True,
    )
    data = res.json()
    return {
        "faithfulness": _clip(data.get("faithfulness")),
        "relevancy": _clip(data.get("relevancy")),
        "correctness": _clip(data.get("correctness")),
        "unsupported_claims": data.get("unsupported_claims", [])[:5],
        "judge_cost_usd": res.cost_usd,
    }
