"""Quality gate for CI - fails the build when a change makes the bot worse.

python -m evals.gate --results evals/results/latest.json
python -m evals.gate --results evals/results/latest.json --update-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASELINE = Path(__file__).resolve().parent / "baseline.json"

# metric -> (direction, allowed change). Relative for "higher is better" metrics.
RULES = {
    "faithfulness": ("higher", 0.05),  # may drop at most 5% relative
    "correctness": ("higher", 0.05),
    "refusal_accuracy": ("higher", 0.05),
    "retrieval_hit_rate": ("higher", 0.05),
    "hallucination_rate": ("lower", 0.02),  # may rise at most 2 points absolute
    "false_refusal_rate": ("lower", 0.05),
}


def check(current: dict, baseline: dict) -> list[str]:
    failures = []
    for metric, (direction, tol) in RULES.items():
        if metric not in baseline or metric not in current:
            continue
        cur, base = float(current[metric]), float(baseline[metric])
        if direction == "higher" and cur < base * (1 - tol):
            failures.append(f"{metric}: {cur:.3f} < baseline {base:.3f} (-{tol:.0%} allowed)")
        if direction == "lower" and cur > base + tol:
            failures.append(f"{metric}: {cur:.3f} > baseline {base:.3f} (+{tol:.2f} allowed)")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="evals/results/latest.json")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()

    current = json.loads(Path(args.results).read_text(encoding="utf-8"))["summary"]
    if args.update_baseline:
        BASELINE.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print(f"baseline updated -> {BASELINE}")
        return 0

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    print(f"{'metric':<22}{'baseline':>10}{'current':>10}")
    for m in RULES:
        if m in baseline and m in current:
            print(f"{m:<22}{baseline[m]:>10.3f}{current[m]:>10.3f}")

    failures = check(current, baseline)
    if failures:
        print("\nQUALITY GATE FAILED")
        for f in failures:
            print("  -", f)
        return 1
    print("\nQUALITY GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
