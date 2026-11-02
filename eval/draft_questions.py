"""
eval/draft_questions.py

Drafts CANDIDATE eval questions from real passages with your local LLM.

v2 change: the evidence span is now picked IN CODE (a real sentence of the chunk), so it is verbatim by
construction. The LLM only writes the question and the short gold answer for that sentence. Drafts are still
auto-rejected when the question does not name the company and the call date, leaks the answer, is vague, or the
gold answer contains a number that is not in the span.

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
    "single_fact": "one specific fact stated in the TARGET (a figure, a percentage, a named product, a decision)",
    "exact_term": "one distinctive term, product or code name, or number that appears in the TARGET, so the "
                  "question can be found by an exact keyword",
    "thematic": "what management said about ONE theme covered by the TARGET (supply, demand, margins, guidance, "
                "competition, capital spending, ...); the answer is one or two sentences",
}
PROMPT = """You write test questions for a search system over earnings call transcripts.
Passage metadata: company = {company}, call date = {date} (fiscal {quarter} {year}).

Below is a PASSAGE and, inside it, a TARGET sentence. Write ONE question of this kind: {kind}.
Rules:
- The question must be answerable from the TARGET and must name the company ("{company}") and the call date
  ("{date}"), for example "On the {company} earnings call of {date}, ...".
- It must be specific and understandable on its own. Name the product, segment or topic; never write
  "what was mentioned about". Use the PASSAGE only to resolve words like "they" or "it" in the TARGET.
- The question must NOT contain the answer.
- Do not ask who said something. Do not mention "the passage", "the target", "the text" or "the transcript".
- The gold answer is short (a few words, or one sentence) and uses only words and numbers found in the TARGET.
Return JSON only: {{"question": "...", "gold_answer": "..."}}

PASSAGE:
{text}

TARGET:
{target}"""

_BAD_QUESTION = re.compile(
    r"\b(who said|who stated|which speaker|the passage|the transcript|the text|the target)\b", re.I)
_VAGUE = re.compile(r"\b(was|were|is|are) mentioned\b|\bmentioned about\b|\bwhat did .* say about\b", re.I)
_BOILERPLATE = re.compile(
    r"forward-looking|safe harbor|non-gaap|\bgaap\b|risk factors|10-k|10-q|press release|webcast|replay|"
    r"conference call|good (morning|afternoon|evening)|thank you|thanks|welcome|next question|operator|"
    r"turn the call|turn it over|let me turn|investor relations|\bslides?\b",
    re.I)
_PRONOUN_START = re.compile(r"^(they|it|its|this|that|these|those|and|but|so|also|which|their|here|there)\b", re.I)
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z$"“(])')
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_TERM = re.compile(r"\b[A-Z][A-Za-z]*\d+[A-Za-z]*\b|\b[A-Z]{2,}\b|\d")  # GB200, NVLink-style, digits


def usable(chunk: dict) -> bool:
    text = chunk["text"]
    return (len(text.split()) >= 60 and chunk.get("role") not in ("operator", "analyst")
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


def _sentence_ok(s: str) -> bool:
    n = len(s.split())
    return 8 <= n <= 45 and "?" not in s and not _BOILERPLATE.search(s) and not _PRONOUN_START.match(s)


def pick_span(text: str, kind: str, rng: random.Random) -> str | None:
    """Pick the evidence span in code: a real sentence (or two, for thematic) copied from the chunk."""
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if kind == "thematic":
        cands = []
        for i in range(len(sents) - 1):
            a, b = sents[i], sents[i + 1]
            if _sentence_ok(a) and "?" not in b and not _BOILERPLATE.search(b):
                start = text.find(a)
                end = text.find(b, start) + len(b) if start >= 0 else -1
                span = text[start:end] if start >= 0 and end > start else ""
                if span and 25 <= len(span.split()) <= 80:
                    cands.append(span)
        return rng.choice(cands) if cands else None
    pattern = _NUM if kind == "single_fact" else _TERM
    cands = [s for s in sents if _sentence_ok(s) and pattern.search(s)]
    return rng.choice(cands) if cands else None


def _numbers(s: str) -> set[str]:
    return {m.rstrip(".,").lstrip("0") or "0" for m in _NUM.findall(s)}


def check_question(r: dict, chunk: dict, company: str, date: str) -> str | None:
    """Return a rejection reason, or None when the draft is acceptable."""
    q = (r.get("question") or "").strip()
    span = (r.get("span") or "").strip()
    gold = (r.get("gold_answer") or "").strip()
    if not q or not span:
        return "empty"
    if not span_in_chunk(chunk["text"], span, threshold=0.95):
        return "span not verbatim"
    if company.lower() not in q.lower() or date not in q:
        return "question does not name company and call date"
    if _BAD_QUESTION.search(q):
        return "asks who said it / mentions the passage"
    if _VAGUE.search(q):
        return "vague question"
    if gold:
        q_nums = _numbers(q.replace(date, ""))
        g_nums = _numbers(gold)
        if g_nums - _numbers(span):
            return "gold answer has a number that is not in the span"
        if g_nums & q_nums:
            return "answer leaked into the question"
        if len(gold) > 3 and gold.lower().rstrip(".") in q.lower():
            return "answer leaked into the question"
    return None


def draft(kind: str, n: int, seed: int, config_name: str = "v2", tickers: set[str] | None = None,
          llm=None, progress=print) -> list[dict]:
    cfg = config.get_chunk_config(config_name)
    dates = {(t["ticker"], t["quarter"], t["year"]): t["date"] for t in queries.get_all_transcripts()}
    pool = [c for c in queries.get_chunks(cfg.hash) if dates.get((c["ticker"], c["quarter"], c["year"]))]
    if not pool:
        sys.exit("No chunks found. Run scripts/run_pipeline.py first.")
    llm = llm or get_llm()
    progress(f"LLM model: {getattr(config, 'LLM_MODEL', '?')} (provider: {getattr(config, 'LLM_PROVIDER', '?')})")
    rng = random.Random(seed + 1000)
    kept: list[dict] = []
    tried = 0
    for c in interleave(pool, seed, tickers=tickers):
        if len(kept) >= n or tried >= n * 6:
            break
        tried += 1
        date = dates[(c["ticker"], c["quarter"], c["year"])]
        company = COMPANY.get(c["ticker"], c["ticker"])
        label = f"{c['ticker']} {c['quarter']} {c['year']}"
        span = pick_span(c["text"], kind, rng)
        if not span:
            progress(f"  [{tried}] {label}: rejected (no suitable sentence)")
            continue
        prompt = PROMPT.format(company=company, date=date, quarter=c["quarter"], year=c["year"],
                               kind=KINDS[kind], text=c["text"], target=span)
        try:
            r = extract_json(llm.chat([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=300,
                                      json_mode=True))
        except LLMError:
            progress(f"  [{tried}] {label}: rejected (model error)")
            continue
        r = {"question": r.get("question"), "gold_answer": r.get("gold_answer"), "span": span}
        why = check_question(r, c, company, date)
        if why:
            progress(f"  [{tried}] {label}: rejected ({why})")
            continue
        kept.append({"id": f"{PREFIX[kind]}{len(kept) + 1:03d}", "category": kind, "question": r["question"].strip(),
                     "gold_answer": (r.get("gold_answer") or "").strip(),
                     "split": "dev" if rng.random() < 0.6 else "test", "needs_human_review": True,
                     "evidence": [{"transcript": f"{c['ticker']}_{c['quarter']}_{c['year']}", "span": span}]})
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