"""
eval/draft_questions.py

Drafts CANDIDATE eval questions from real passages with your local LLM, then auto-rejects every draft whose
evidence span is not a verbatim substring of the passage, or whose question does not name the company and the call
date (the retrieval eval runs without filters, so a question must identify its call by itself).

Output is a starting point only: you must read each one, fix the gold answer and keep it only if a human agrees.

    python -m eval.draft_questions --kind single_fact --n 24 --seed 1
    python -m eval.draft_questions --kind exact_term  --n 16 --seed 2
    python -m eval.draft_questions --kind thematic    --n 16 --seed 3

Writes eval/cache/candidates_<kind>.jsonl (git-ignored: it holds verbatim transcript spans).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import span_in_chunk  # noqa: E402
from src import config  # noqa: E402
from src.database import queries  # noqa: E402
from src.llm import LLMError, extract_json, get_llm  # noqa: E402

COMPANY = {"AAPL": "Apple", "AMZN": "Amazon", "GOOGL": "Alphabet", "MSFT": "Microsoft", "NVDA": "NVIDIA"}
PREFIX = {"single_fact": "sf", "exact_term": "et", "thematic": "th"}
KINDS = {
    "single_fact": "one specific fact stated in the passage (a figure, a percentage, a named product, a decision)",
    "exact_term": "one distinctive term, product or code name, or number that appears in the passage, so the question "
                  "can be found by an exact keyword",
    "thematic": "what management said about ONE theme (supply, demand, margins, guidance, competition, capital "
                "spending, ...); the answer is one or two sentences",
}
PROMPT = """You write test questions for a search system over earnings call transcripts.
Passage metadata: company = {company}, call date = {date} (fiscal {quarter} {year}).

Read the passage and write ONE question of this kind: {kind}.
Rules:
- The question must name the company ("{company}") and the call date ("{date}"), for example
  "On the {company} earnings call of {date}, ...".
- It must be answerable from this passage alone, and be specific.
- Do not ask who said something. Do not mention "the passage", "the text" or "the transcript".
- Give the short gold answer, and the exact sentence from the passage that contains it, copied word for word.
Return JSON only: {{"question": "...", "gold_answer": "...", "span": "..."}}

PASSAGE:
{text}"""
_BAD_QUESTION = re.compile(r"\b(who said|who stated|which speaker|the passage|the transcript|the text)\b", re.I)


def usable(chunk: dict) -> bool:
    text = chunk["text"]
    return (len(text.split()) >= 60 and chunk.get("role") != "operator"
            and "cold in space" not in text)  # NVDA_Q4_2026 line 252 is cut off mid-sentence


def interleave(chunks: list[dict], seed: int, per_call: int = 2, tickers: set[str] | None = None) -> list[dict]:
    """Shuffle, cap the number per call, and alternate companies so every company is covered."""
    rng = random.Random(seed)
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        if usable(c) and (not tickers or c["ticker"] in tickers):
            by_ticker[c["ticker"]].append(c)
    for lst in by_ticker.values():
        rng.shuffle(lst)
    used: dict[tuple, int] = defaultdict(int)
    out: list[dict] = []
    queues = [lst for _, lst in sorted(by_ticker.items())]
    while any(queues):
        for q in queues:
            while q:
                c = q.pop()
                key = (c["ticker"], c["quarter"], c["year"])
                if used[key] < per_call:
                    used[key] += 1
                    out.append(c)
                    break
    return out


def check_question(r: dict, chunk: dict, company: str, date: str) -> str | None:
    """Return a rejection reason, or None when the draft is acceptable."""
    q = (r.get("question") or "").strip()
    span = (r.get("span") or "").strip()
    if not q or not span:
        return "empty"
    if not span_in_chunk(chunk["text"], span, threshold=0.95):
        return "span not verbatim"
    if company.lower() not in q.lower() or date not in q:
        return "question does not name company and call date"
    if _BAD_QUESTION.search(q):
        return "asks who said it / mentions the passage"
    return None


def draft(kind: str, n: int, seed: int, config_name: str = "v2", tickers: set[str] | None = None,
          llm=None, progress=print) -> list[dict]:
    cfg = config.get_chunk_config(config_name)
    dates = {(t["ticker"], t["quarter"], t["year"]): t["date"] for t in queries.get_all_transcripts()}
    pool = [c for c in queries.get_chunks(cfg.hash) if dates.get((c["ticker"], c["quarter"], c["year"]))]
    if not pool:
        sys.exit("No chunks found. Run scripts/run_pipeline.py first.")
    llm = llm or get_llm()
    rng = random.Random(seed + 1000)
    kept: list[dict] = []
    tried = 0
    for c in interleave(pool, seed, tickers=tickers):
        if len(kept) >= n or tried >= n * 4:
            break
        tried += 1
        date = dates[(c["ticker"], c["quarter"], c["year"])]
        company = COMPANY.get(c["ticker"], c["ticker"])
        label = f"{c['ticker']} {c['quarter']} {c['year']}"
        prompt = PROMPT.format(company=company, date=date, quarter=c["quarter"], year=c["year"],
                               kind=KINDS[kind], text=c["text"])
        try:
            r = extract_json(llm.chat([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=300,
                                      json_mode=True))
        except LLMError:
            progress(f"  [{tried}] {label}: rejected (model error)")
            continue
        why = check_question(r, c, company, date)
        if why:
            progress(f"  [{tried}] {label}: rejected ({why})")
            continue
        kept.append({"id": f"{PREFIX[kind]}{len(kept) + 1:03d}", "category": kind, "question": r["question"].strip(),
                     "gold_answer": (r.get("gold_answer") or "").strip(),
                     "split": "dev" if rng.random() < 0.6 else "test", "needs_human_review": True,
                     "evidence": [{"transcript": f"{c['ticker']}_{c['quarter']}_{c['year']}",
                                   "span": r["span"].strip()}]})
        progress(f"  [{tried}] {label}: kept ({len(kept)}/{n})")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=list(KINDS), default="single_fact")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--config", default="v2")
    ap.add_argument("--tickers", nargs="+", help="only these tickers, e.g. --tickers NVDA AAPL")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or config.EVAL_DIR / "cache" / f"candidates_{args.kind}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = draft(args.kind, args.n, args.seed, args.config, {t.upper() for t in args.tickers or []} or None)
    out.write_text("\n".join(json.dumps(k) for k in kept) + ("\n" if kept else ""), encoding="utf-8")
    print(f"\nkept {len(kept)} candidate(s) -> {out}\nNow review each one by hand: fix gold_answer, drop weak questions.")


if __name__ == "__main__":
    main()
