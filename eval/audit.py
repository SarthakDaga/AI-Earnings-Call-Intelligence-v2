"""Blind hand-audit of the LLM judge.

Step 1 - print a sheet of N answers (judge score hidden), judge them yourself:
    python -m eval.audit --row B1 --n 10 --seed 7
Step 2 - enter your verdicts (1 = correct, 0 = wrong, in the order printed) to compare with the judge:
    python -m eval.audit --row B1 --n 10 --seed 7 --verdicts 1 0 1 1 0 1 1 1 0 1

Sampled from answered (not abstained), answerable core questions (single_fact, exact_term, thematic).
"Correct" means: the answer states the gold fact(s) and gives no contradicting figure.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--chars", type=int, default=600)
    ap.add_argument("--verdicts", nargs="*", type=int)
    args = ap.parse_args()
    files = sorted((config.EVAL_DIR / "results").glob("*-answers.json"))
    if not files:
        sys.exit("no answers results found")
    res = json.loads(files[-1].read_text(encoding="utf-8"))
    rows = [r for r in res["rows"][args.row]["per_question"]
            if r["answerable"] and not r["agent_subset"] and not r["abstained"]]
    rows.sort(key=lambda r: r["id"])
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.n]
    if not args.verdicts:
        print(f"run {res['run_id']}  row {args.row}: {len(rows)} sampled answers (judge hidden)\n")
        for i, r in enumerate(rows, 1):
            print(f"[{i}] {r['id']} {r['category']}\n  Q:    {r['question']}\n  GOLD: {r['gold']}\n"
                  f"  ANS:  {(r['answer'] or '').replace(chr(10), ' ')[:args.chars]}\n")
        print("Judge each one (1 = correct, 0 = wrong), then rerun with --verdicts <your list in order>.")
        return
    if len(args.verdicts) != len(rows):
        sys.exit(f"expected {len(rows)} verdicts, got {len(args.verdicts)}")
    agree = 0
    for i, (r, v) in enumerate(zip(rows, args.verdicts, strict=True), 1):
        judge = 1 if (r["correctness"] or 0.0) >= 0.5 else 0
        agree += judge == v
        print(f"[{i}] {r['id']}: you={v} judge={judge} (correctness={r['correctness']}) "
              f"{'' if judge == v else '<-- DISAGREE'}")
    print(f"\nagreement: {agree}/{len(rows)}")


if __name__ == "__main__":
    main()
