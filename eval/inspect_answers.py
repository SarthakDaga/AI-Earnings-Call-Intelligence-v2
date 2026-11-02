"""Print a per-question view of the latest answers eval, so you can see WHY correctness is low.

    python -m eval.inspect_answers            (latest answers run)
    python -m eval.inspect_answers --row B1 --only-wrong
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", default=None, help="B0, B1 or B2 (default: every row in the file)")
    ap.add_argument("--only-wrong", action="store_true", help="only answerable questions with correctness < 0.5")
    ap.add_argument("--chars", type=int, default=150)
    args = ap.parse_args()

    files = sorted((config.EVAL_DIR / "results").glob("*-answers.json"))
    if not files:
        sys.exit("no answers results found")
    res = json.loads(files[-1].read_text(encoding="utf-8"))
    print(f"run {res['run_id']} split={res['split']}")
    for key, block in res["rows"].items():
        if args.row and key != args.row:
            continue
        rows = block["per_question"]
        ans = [r for r in rows if r["answerable"]]
        abst = [r for r in ans if r["abstained"]]
        print(f"\n=== {key} {block['label']}: {len(rows)} questions, {len(ans)} answerable, "
              f"{len(abst)} of those abstained (scored 0)")
        cat: dict[str, list[float]] = defaultdict(list)
        for r in ans:
            cat[r["category"]].append(r["correctness"] if r["correctness"] is not None else 0.0)
        for c, v in sorted(cat.items()):
            print(f"   {c:14s} n={len(v):2d} mean correctness {sum(v) / len(v):.2f}")
        print()
        for r in rows:
            corr = r["correctness"]
            if args.only_wrong and not (r["answerable"] and (corr is None or corr < 0.5)):
                continue
            flag = "ABSTAINED" if r["abstained"] else "answered"
            print(f"{r['id']} [{r['category']}] {flag} correct={corr} faith={r['faithfulness']} "
                  f"cit={r['citation_precision']}")
            print(f"   Q:    {r['question'][:args.chars]}")
            print(f"   GOLD: {(r['gold'] or '')[:args.chars]}")
            print(f"   ANS:  {(r['answer'] or '').replace(chr(10), ' ')[:args.chars]}")


if __name__ == "__main__":
    main()
