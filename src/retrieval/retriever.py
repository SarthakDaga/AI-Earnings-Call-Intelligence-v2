"""
src/retrieval/retriever.py

One retriever interface (R2.6) used identically by the API, the agent and the
eval harness:

    mode = dense | bm25 | hybrid | hybrid_rerank

Pipeline for hybrid_rerank:
    dense top-50 + BM25 top-50 -> RRF (k=60) keep 30 -> cross-encoder -> top-k
Metadata filters (ticker, quarter, year, section, role) apply to both legs (R2.7).
"""

from __future__ import annotations

import time

from src import config
from src.database import queries
from src.preprocessing.chunker import Chunk, context_header
from src.retrieval import dense
from src.retrieval.bm25 import BM25Index
from src.retrieval.embedder import Embedder
from src.retrieval.fusion import rrf
from src.retrieval.reranker import CrossEncoderReranker

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")
FILTER_KEYS = ("ticker", "quarter", "year", "section", "role")


class Retriever:
    def __init__(self, cfg: config.ChunkConfig | None = None, embedder=None, reranker=None, chroma_path=None):
        self.cfg = cfg or config.get_chunk_config()
        self.embedder = embedder or Embedder(self.cfg.embed_model)
        self._reranker = reranker
        self.chroma_path = chroma_path
        self.chunks: dict[str, dict] = {}
        self.bm25: BM25Index | None = None
        self.rerank_error: str | None = None  # set when hybrid_rerank had to fall back to hybrid

    # ── lifecycle ──
    def ensure_ready(self) -> "Retriever":
        if self.bm25 is None:
            rows = queries.get_chunks(self.cfg.hash)
            self.chunks = {r["chunk_id"]: r for r in rows}
            self.bm25 = BM25Index(rows)
        return self

    def reload(self) -> None:
        self.bm25 = None
        self.ensure_ready()

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(config.RERANKER_MODEL)
        return self._reranker

    def status(self) -> dict:
        return {"chunks": len(self.chunks), "bm25_ready": self.bm25 is not None,
                "embedder_loaded": bool(getattr(self.embedder, "loaded", False)),
                "reranker_loaded": bool(self._reranker is not None and getattr(self._reranker, "loaded", False)),
                "reranker_error": self.rerank_error, "config": self.cfg.name}

    # ── helpers ──
    @staticmethod
    def _clean_filters(filters: dict | None) -> dict:
        f = {k: v for k, v in (filters or {}).items() if k in FILTER_KEYS and v not in (None, "")}
        if "ticker" in f:
            f["ticker"] = str(f["ticker"]).upper()
        if "quarter" in f:
            f["quarter"] = str(f["quarter"]).upper()
        if "year" in f:
            f["year"] = int(f["year"])
        return f

    @staticmethod
    def _allow(filters: dict):
        if not filters:
            return None
        return lambda c: all(c.get(k) == v for k, v in filters.items())

    def _header(self, c: dict) -> str:
        if not self.cfg.context_header:
            return ""
        return context_header(c["ticker"], c["quarter"], c["year"],
                              Chunk(c["text"], c.get("speaker") or "", c.get("role") or "",
                                    c.get("section") or "")) + "\n"

    def _hit(self, cid: str, score: float, dense_rank=None, bm25_rank=None) -> dict:
        c = self.chunks[cid]
        return {"chunk_id": cid, "text": c["text"], "ticker": c["ticker"], "quarter": c["quarter"],
                "year": c["year"], "section": c.get("section") or "", "speaker": c.get("speaker") or "",
                "role": c.get("role") or "", "transcript_id": c["transcript_id"],
                "chunk_index": c.get("chunk_index"), "score": round(float(score), 4),
                "dense_rank": dense_rank, "bm25_rank": bm25_rank}

    # ── search ──
    def search(self, query: str, k: int = 5, mode: str = "hybrid_rerank", filters: dict | None = None,
               dense_k: int = 50, bm25_k: int = 50, fuse_k: int = 30) -> list[dict]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.ensure_ready()
        if not self.chunks:
            return []
        f = self._clean_filters(filters)

        dense_hits: list[tuple[str, float]] = []
        bm25_hits: list[tuple[str, float]] = []
        if mode in ("dense", "hybrid", "hybrid_rerank"):
            qv = self.embedder.encode_query(query)
            dense_hits = [(i, s) for i, s in dense.search(self.cfg, qv, dense_k, f, self.chroma_path)
                          if i in self.chunks]
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            bm25_hits = self.bm25.search(query, bm25_k, self._allow(f))

        d_rank = {cid: r for r, (cid, _) in enumerate(dense_hits, 1)}
        b_rank = {cid: r for r, (cid, _) in enumerate(bm25_hits, 1)}

        if mode == "dense":
            return [self._hit(cid, s, d_rank[cid]) for cid, s in dense_hits[:k]]
        if mode == "bm25":
            return [self._hit(cid, s, None, b_rank[cid]) for cid, s in bm25_hits[:k]]

        fused = rrf([[c for c, _ in dense_hits], [c for c, _ in bm25_hits]], k=60)
        if mode == "hybrid":
            return [self._hit(cid, s, d_rank.get(cid), b_rank.get(cid)) for cid, s in fused[:k]]

        candidates = fused[:fuse_k]
        texts = [self._header(self.chunks[cid]) + self.chunks[cid]["text"] for cid, _ in candidates]
        try:
            if getattr(self.reranker, "loading", False):  # still downloading in the background: don't block
                raise RuntimeError("reranker is still loading")
            scores = self.reranker.score(query, texts)
        except Exception as e:  # model missing / download failed / offline: degrade, don't fail the search
            if not self.rerank_error:
                print(f"[Retriever] Reranker unavailable ({type(e).__name__}: {str(e)[:120]}); "
                      f"falling back to hybrid order.")
            self.rerank_error = f"{type(e).__name__}: {str(e)[:200]}"
            return [self._hit(cid, s, d_rank.get(cid), b_rank.get(cid)) for cid, s in fused[:k]]
        self.rerank_error = None
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:k]
        return [self._hit(cid, sc, d_rank.get(cid), b_rank.get(cid)) for (cid, _), sc in ranked]

    def timed_search(self, *args, **kwargs) -> tuple[list[dict], float]:
        t0 = time.perf_counter()
        hits = self.search(*args, **kwargs)
        return hits, (time.perf_counter() - t0) * 1000
