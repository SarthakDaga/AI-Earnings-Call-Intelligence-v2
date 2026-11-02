"""Re-aggregate the latest answers run by split (dev / test / all) with bootstrap CIs, plus paired B1-B0 / B2-B1
differences on the same questions. No model calls; reads eval/results/*-answers.json.

    python -m eval.resplit
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import bootstrap_ci  # noqa: E402
from src import config  # noqa: E402


def core(rows, split):
    return [r for r in rows if r["answerable"] and not r["agent_subset"] and (split == "all" or r["split"] == split)]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 2) if xs else None


def paired(a_rows, b_rows, split):
    a = {r["id"]: r["correctness"] or 0.0 for r in core(a_rows, split)}
    b = {r["id"]: r["correctness"] or 0.0 for r in core(b_rows, split)}
    d = [b[i] - a[i] for i in a if i in b]
    if not d:
        return None
    rng = random.Random(0)
    means = sorted(statistics.fmean(rng.choices(d, k=len(d))) for _ in range(2000))
    return len(d), round(statistics.fmean(d), 2), [round(means[50], 2), round(means[1949], 2)]


def main() -> None:
    files = sorted((config.EVAL_DIR / "results").glob("*-answers.json"))
    if not files:
        sys.exit("no answers results found")
    res = json.loads(files[-1].read_text(encoding="utf-8"))
    print(f"run {res['run_id']}  (core = single_fact + exact_term + thematic)\n")
    rows = {k: v["per_question"] for k, v in res["rows"].items()}
    print(f"{'system':8s} {'split':5s} {'n':>3s} {'correct':>8s} {'95% CI':>14s} {'faith':>6s} {'cit.prec':>9s} "
          f"{'abstain(unans.)':>16s}")
    for key, rs in rows.items():
        for split in ("dev", "test", "all"):
            c = core(rs, split)
            un = [r for r in rs if not r["answerable"] and (split == "all" or r["split"] == split)]
            ci = bootstrap_ci([r["correctness"] for r in c])
            ab = mean([float(r["abstained"]) for r in un])
            print(f"{key:8s} {split:5s} {len(c):3d} {mean([r['correctness'] for r in c])!s:>8s} {ci!s:>14s} "
                  f"{mean([r['faithfulness'] for r in c])!s:>6s} {mean([r['citation_precision'] for r in c])!s:>9s} "
                  f"{ab!s:>10s} (n={len(un)})")
    print("\npaired difference in correctness on the same questions (mean, 95% CI):")
    for a, b in (("B0", "B1"), ("B1", "B2"), ("B0", "B2")):
        if a in rows and b in rows:
            for split in ("test", "all"):
                p = paired(rows[a], rows[b], split)
                if p:
                    print(f"  {b} - {a} [{split}] n={p[0]}: {p[1]:+.2f}  CI {p[2]}")


if __name__ == "__main__":
    main()
