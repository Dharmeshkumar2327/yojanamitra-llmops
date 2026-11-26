"""Grow the golden set from your own indexed chunks (synthetic test generation).

    python -m evals.generate_synthetic --n 80 --out evals/synthetic_candidates.jsonl

IMPORTANT: generated items are CANDIDATES. Review each one by hand (read the source chunk),
fix or delete bad ones, then append the good ones to golden_dataset.jsonl.
A golden set is only as good as your review.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

GEN_PROMPT = """From the passage below about an Indian Government scheme, write ONE question a citizen
might ask and its exact answer using only the passage. Question style: {style}.
Return JSON only: {{"question": str, "answer": str}}

Scheme: {scheme} | Section: {section}
Passage:
{text}"""

STYLES = [
    "simple factual question in English",
    "eligibility question describing a person's situation in English",
    "simple question in Hindi (Devanagari), answer in Hindi",
    "question with a WRONG assumption (wrong amount or age) that the answer must correct",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--out", default="evals/synthetic_candidates.jsonl")
    args = ap.parse_args()

    from yojanamitra.config import get_settings
    from yojanamitra.generation.llm import get_llm

    s = get_settings()
    llm = get_llm()
    chunks = [
        json.loads(x)
        for x in Path(args.chunks).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    chunks = [c for c in chunks if len(c["raw_text"].split()) > 40]
    random.seed(42)
    sample = random.sample(chunks, min(args.n, len(chunks)))

    with open(args.out, "w", encoding="utf-8") as f:
        for i, c in enumerate(sample):
            style = STYLES[i % len(STYLES)]
            res = llm.chat(
                [
                    {
                        "role": "user",
                        "content": GEN_PROMPT.format(
                            style=style,
                            scheme=c["scheme_name"],
                            section=c["section"],
                            text=c["raw_text"],
                        ),
                    }
                ],
                model=s.llm_model,
                temperature=0.4,
                max_tokens=300,
                json_mode=True,
            )
            data = res.json()
            if not data.get("question"):
                continue
            typ = (
                "hindi"
                if "Hindi" in style
                else "false_premise"
                if "WRONG" in style
                else "eligibility"
                if "eligibility" in style
                else "factual"
            )
            f.write(
                json.dumps(
                    {
                        "id": f"syn{i:03d}",
                        "question": data["question"],
                        "reference_answer": data["answer"],
                        "expected_scheme": c["scheme_id"],
                        "type": typ,
                        "expect_refusal": False,
                        "core": False,
                        "source_chunk_id": c["id"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote candidates -> {args.out}  (REVIEW before adding to golden_dataset.jsonl)")


if __name__ == "__main__":
    main()
