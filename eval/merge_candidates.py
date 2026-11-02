"""
eval/merge_candidates.py

Moves the candidates you approved from a drafts file into the real question set:
  * the question goes to eval/questions.jsonl (public, no transcript text),
  * the verbatim evidence span goes to eval/evidence.private.jsonl (git-ignored),
  * ids are renumbered q101, q102, ... so they never clash with the starter questions.

    python -m eval.merge_candidates eval\\cache\\candidates_single_fact.jsonl --keep sf001 sf004 sf007

Edit question / gold_answer / split in the candidates file BEFORE merging. Each id can only be merged once.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


def read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]


def merge(cand_path: Path, keep: list[str], questions: Path, private: Path) -> list[str]:
    cands = {c["id"]: c for c in read_lines(cand_path)}
    missing = [k for k in keep if k not in cands]
    if missing:
        raise SystemExit(f"not in {cand_path.name}: {missing}")
    existing = read_lines(questions)
    taken = {q["question"].strip().lower() for q in existing}
    numbers = [int(m.group(1)) for q in existing if (m := re.fullmatch(r"q(\d+)", q["id"]))]
    nxt = max([101, *[n + 1 for n in numbers]])
    new_ids, q_lines, ev_lines = [], [], []
    for cid in keep:
        c = cands[cid]
        if c["question"].strip().lower() in taken:
            print(f"skipped {cid}: this question is already in {questions.name}")
            continue
        taken.add(c["question"].strip().lower())
        qid = f"q{nxt:03d}"
        nxt += 1
        q_lines.append({"id": qid, "category": c["category"], "question": c["question"].strip(),
                        "gold_answer": c["gold_answer"].strip(), "evidence": [], "split": c.get("split", "dev")})
        ev_lines.append({"id": qid, "evidence": c["evidence"]})
        new_ids.append(qid)
    with questions.open("a", encoding="utf-8") as f:
        f.writelines(json.dumps(q) + "\n" for q in q_lines)
    with private.open("a", encoding="utf-8") as f:
        f.writelines(json.dumps(e) + "\n" for e in ev_lines)
    return new_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--keep", nargs="+", required=True, help="candidate ids to keep, e.g. sf001 sf004")
    ap.add_argument("--questions", default=str(config.EVAL_DIR / "questions.jsonl"))
    ap.add_argument("--private", default=str(config.EVAL_DIR / "evidence.private.jsonl"))
    a = ap.parse_args()
    ids = merge(Path(a.candidates), a.keep, Path(a.questions), Path(a.private))
    print(f"merged {len(ids)} question(s): {', '.join(ids)}")
    print("Next: python -m eval.verify_evidence")


if __name__ == "__main__":
    main()
