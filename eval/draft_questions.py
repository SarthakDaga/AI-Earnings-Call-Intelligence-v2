"""
eval/draft_questions.py

Drafts CANDIDATE questions from random passages using your free local LLM, then
auto-rejects any whose evidence span is not a verbatim substring of the passage.
Output is a starting point only: you must read each one, fix the gold answer
and keep it only if a human agrees. Questions the model writes unchecked are
not a benchmark.

    python -m eval.draft_questions --n 40 --seed 1   ->  eval/candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import span_in_chunk  # noqa: E402
from src import config  # noqa: E402
from src.database import queries  # noqa: E402
from src.llm import LLMError, extract_json, get_llm  # noqa: E402

PROMPT = """Read this earnings call passage. Write ONE specific question that this passage answers, the short answer,
and the exact sentence from the passage that contains the answer (copied verbatim).
Avoid vague questions. Include the topic or number in the question so it can be searched.
Return JSON only: {"question": "...", "gold_answer": "...", "span": "..."}

PASSAGE:
%s"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="v2")
    ap.add_argument("--out", default=str(config.EVAL_DIR / "candidates.jsonl"))
    args = ap.parse_args()

    cfg = config.get_chunk_config(args.config)
    pool = [c for c in queries.get_chunks(cfg.hash) if len(c["text"].split()) >= 60]
    if not pool:
        sys.exit("No chunks found. Run scripts/run_pipeline.py first.")
    random.Random(args.seed).shuffle(pool)
    llm, kept, rejected = get_llm(), [], 0
    for c in pool[: args.n * 3]:
        if len(kept) >= args.n:
            break
        try:
            r = extract_json(llm.chat([{"role": "user", "content": PROMPT % c["text"]}],
                                      temperature=0.3, max_tokens=300, json_mode=True))
        except LLMError:
            rejected += 1
            continue
        span = (r.get("span") or "").strip()
        if not r.get("question") or not span_in_chunk(c["text"], span, threshold=0.95):
            rejected += 1
            continue
        kept.append({"id": f"cand{len(kept) + 1:03d}", "category": "TODO", "question": r["question"],
                     "gold_answer": r.get("gold_answer", ""), "split": "TODO", "needs_human_review": True,
                     "evidence": [{"transcript": f"{c['ticker']}_{c['quarter']}_{c['year']}", "span": span}]})
    Path(args.out).write_text("\n".join(json.dumps(k) for k in kept) + "\n")
    print(f"kept {len(kept)}, auto-rejected {rejected} (span not verbatim / unparseable). Now review by hand -> {args.out}")


if __name__ == "__main__":
    main()
