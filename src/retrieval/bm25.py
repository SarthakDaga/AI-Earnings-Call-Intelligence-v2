"""
src/retrieval/bm25.py

In-memory BM25 with a finance-aware tokenizer (R2.1): keeps $26.0, 26%, Q3,
FY25 and hyphenated terms intact, no stemming. Built from the SQLite chunk
table at startup (takes seconds at this scale).
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from src.ingestion.turns import normalize_numbers

_TOKEN_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?|[A-Za-z][A-Za-z0-9&'\-]*")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were",
         "be", "that", "this", "it", "as", "at", "by", "with", "we", "our", "you", "i", "so", "but"}


def tokenize(text: str) -> list[str]:
    text = normalize_numbers(text)  # make '$68,000,000,000' match a '$68 billion' query
    toks = []
    for t in _TOKEN_RE.findall(text.lower()):
        t = t.rstrip(",").replace("'s", "")
        if t and t not in _STOP:
            toks.append(t)
    return toks


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks]) if chunks else None

    def search(self, query: str, k: int = 50, allow=None) -> list[tuple[str, float]]:
        """allow: optional predicate(chunk_dict)->bool for metadata filters."""
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            if allow is not None and not allow(self.chunks[i]):
                continue
            out.append((self.chunks[i]["chunk_id"], float(scores[i])))
            if len(out) >= k:
                break
        return out
