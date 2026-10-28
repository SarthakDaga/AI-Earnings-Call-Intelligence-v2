"""
eval/judges.py

LLM-as-judge (WS1.6). Free by default: a local Ollama model from a DIFFERENT
family than the generator (JUDGE_MODEL, default qwen2.5:7b vs llama3.2:3b),
temperature 0, prompts versioned below, every judgment cached to disk so reruns
cost nothing and results are reproducible. Use `audit_sample` to hand-check
judgments and report agreement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src import config
from src.llm.provider import LLMError, extract_json, get_llm

PROMPT_VERSION = "v1"
CACHE_DIR = config.EVAL_DIR / "cache" / "judge"

FAITHFULNESS_PROMPT = """You are a strict fact-checker. Judge ONLY against the CONTEXT, never outside knowledge.
1. Split the ANSWER into distinct factual claims.
2. For each claim decide if the CONTEXT directly supports it.
Return JSON only: {"claims": [{"claim": "...", "supported": true or false}]}
CONTEXT:
%(context)s

ANSWER:
%(answer)s"""

CORRECTNESS_PROMPT = """Compare the ANSWER to the GOLD answer for the QUESTION.
Score 1 if it conveys the same key facts, 0.5 if partially correct or missing a key part, 0 if wrong or irrelevant.
Return JSON only: {"score": 0 or 0.5 or 1, "reason": "one short sentence"}
QUESTION: %(question)s
GOLD: %(gold)s
ANSWER: %(answer)s"""


class Judge:
    def __init__(self, llm=None, cache_dir: Path | None = None):
        self.llm = llm or get_llm(role="judge")
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.failures = 0

    def _key(self, prompt: str) -> Path:
        h = hashlib.sha256(f"{PROMPT_VERSION}|{self.llm.model}|{prompt}".encode()).hexdigest()[:24]
        return self.cache_dir / f"{h}.json"

    def _ask(self, prompt: str) -> dict | None:
        f = self._key(prompt)
        if f.exists():
            return json.loads(f.read_text())
        try:
            out = extract_json(self.llm.chat([{"role": "user", "content": prompt}], temperature=0.0,
                                             max_tokens=700, json_mode=True))
        except LLMError:
            self.failures += 1
            return None
        f.write_text(json.dumps({**out, "_judge_model": self.llm.model, "_prompt_version": PROMPT_VERSION}))
        return out

    def faithfulness(self, answer: str, context: str) -> float | None:
        r = self._ask(FAITHFULNESS_PROMPT % {"context": context[:9000], "answer": answer})
        claims = (r or {}).get("claims") or []
        if not claims:
            return None
        return sum(1 for c in claims if c.get("supported") is True) / len(claims)

    def correctness(self, question: str, gold: str, answer: str) -> float | None:
        r = self._ask(CORRECTNESS_PROMPT % {"question": question, "gold": gold, "answer": answer})
        try:
            s = float((r or {}).get("score"))
        except (TypeError, ValueError):
            return None
        return min((0.0, 0.5, 1.0), key=lambda x: abs(x - s))


def audit_sample(records: list[dict], n: int = 10, seed: int = 0, out: Path | None = None) -> Path:
    """Write n random judged items to CSV for hand-labelling. Add a `human` column, then compute agreement."""
    import csv
    import random
    rows = random.Random(seed).sample(records, min(n, len(records)))
    out = Path(out or config.EVAL_DIR / "results" / "judge_audit.csv")
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "gold", "answer", "judge_correctness", "judge_faithfulness", "human"])
        for r in rows:
            w.writerow([r["id"], r["question"], r["gold"], r["answer"], r.get("correctness"), r.get("faithfulness"), ""])
    return out


def agreement(path: Path) -> float:
    import csv
    rows = [r for r in csv.DictReader(open(path)) if r["human"].strip()]
    if not rows:
        return float("nan")
    ok = sum(abs(float(r["human"]) - float(r["judge_correctness"] or 0)) < 0.25 for r in rows)
    return ok / len(rows)
