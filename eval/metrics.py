"""
eval/metrics.py

Retrieval + answer metrics (WS1.5). Evidence is stored as verbatim TEXT SPANS,
not chunk ids, so the same question set works across every chunking strategy.
A retrieved chunk is relevant if it (a) belongs to the evidence transcript and
(b) contains the normalized span, or covers >= 80% of the span's tokens when a
chunk boundary splits it.
"""

from __future__ import annotations

import difflib
import json
import random
import re
import statistics
from pathlib import Path

from src import config
from src.ingestion.turns import normalize_numbers

KS = (1, 3, 5, 10)
AGENT_CATEGORIES = {"cross_quarter", "price_linked"}
CATEGORIES = ("single_fact", "exact_term", "thematic", "cross_quarter", "price_linked", "unanswerable", "trap")


def norm_text(s: str) -> str:
    s = normalize_numbers(s).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def span_in_chunk(chunk_text: str, span: str, threshold: float = 0.8) -> bool:
    c, s = norm_text(chunk_text), norm_text(span)
    if not s:
        return False
    if s in c:
        return True
    st, ct = s.split(), c.split()
    if len(st) < 5:
        return False
    # fuzzy path: every number in the span must appear in the chunk (a different figure is a different fact)
    if not {t for t in st if any(ch.isdigit() for ch in t)} <= set(ct):
        return False
    m = difflib.SequenceMatcher(None, ct, st, autojunk=False)
    matched = sum(b.size for b in m.get_matching_blocks())
    return matched / len(st) >= threshold


def hit_matches(hit: dict, evidence: dict) -> bool:
    key = f"{hit['ticker']}_{hit['quarter']}_{hit['year']}"
    if evidence.get("transcript") and evidence["transcript"] != key:
        return False
    return span_in_chunk(hit["text"], evidence["span"])


def hit_relevant(hit: dict, evidence_list: list[dict]) -> bool:
    return any(hit_matches(hit, e) for e in evidence_list)


# ── per-question retrieval scoring ──
def score_retrieval(hits: list[dict], evidence_list: list[dict], ks=KS) -> dict:
    rel = [hit_relevant(h, evidence_list) for h in hits]
    out = {}
    for k in ks:
        out[f"hit@{k}"] = float(any(rel[:k]))
        covered = sum(1 for e in evidence_list if any(hit_matches(h, e) for h in hits[:k]))
        out[f"cov@{k}"] = covered / len(evidence_list) if evidence_list else 0.0
    first = next((i for i, r in enumerate(rel[:10], 1) if r), None)
    out["rr"] = 1.0 / first if first else 0.0
    return out


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0, alpha: float = 0.05):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(vals, k=len(vals))) for _ in range(n))
    return [round(means[int(n * alpha / 2)], 3), round(means[int(n * (1 - alpha / 2)) - 1], 3)]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def aggregate_retrieval(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    agg = {"n": len(rows)}
    for k in KS:
        agg[f"recall@{k}"] = round(statistics.fmean(r[f"hit@{k}"] for r in rows), 3)
        agg[f"coverage@{k}"] = round(statistics.fmean(r[f"cov@{k}"] for r in rows), 3)
    agg["mrr"] = round(statistics.fmean(r["rr"] for r in rows), 3)
    lat = [r["latency_ms"] for r in rows]
    agg["p50_ms"], agg["p95_ms"] = round(percentile(lat, 50), 1), round(percentile(lat, 95), 1)
    agg["ci95_recall@5"] = bootstrap_ci([r["hit@5"] for r in rows])
    agg["ci95_mrr"] = bootstrap_ci([r["rr"] for r in rows])
    return agg


def by_category(rows: list[dict], aggregator) -> dict:
    cats = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    return {c: aggregator(v) for c, v in sorted(cats.items())}


# ── question loading ──
def load_questions(path: Path | None = None, split: str = "all") -> list[dict]:
    path = Path(path or config.EVAL_DIR / "questions.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See eval/README.md to build the question set "
                                f"(or copy eval/questions.example.jsonl).")
    qs = [json.loads(l) for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    private = path.parent / "evidence.private.jsonl"
    if private.exists():  # evidence spans may be kept out of the public repo (licensing)
        ev = {}
        for l in private.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                ev[r["id"]] = r["evidence"]
        for q in qs:
            q.setdefault("evidence", ev.get(q["id"], []))
            if not q["evidence"]:
                q["evidence"] = ev.get(q["id"], [])
    for q in qs:
        q.setdefault("evidence", [])
        q.setdefault("answerable", q.get("category") != "unanswerable")
        q.setdefault("expected_tools", [])
        q.setdefault("split", "dev")
        q["agent_subset"] = q.get("category") in AGENT_CATEGORIES
    if split != "all":
        qs = [q for q in qs if q["split"] == split]
    return qs


# ── answer metrics helpers ──
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> list[str]:
    return [n.replace(",", "").rstrip(".") for n in _NUM.findall(normalize_numbers(text))]


def numeric_exact_ok(gold: str, answer: str) -> bool:
    """All numbers in the gold answer must appear (as whole numbers) in the answer."""
    ans = normalize_numbers(answer).replace(",", "")
    return all(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d]|\.\d)", ans) for n in numbers_in(gold))


def citation_precision(cited_hits: list[dict], evidence_list: list[dict]) -> float | None:
    if not cited_hits:
        return None
    return sum(hit_relevant(h, evidence_list) for h in cited_hits) / len(cited_hits)


def tool_selection_ok(expected: list[str], used: list[str]) -> bool:
    return set(expected) <= set(used)
