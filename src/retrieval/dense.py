"""
src/retrieval/dense.py
ChromaDB dense store. The collection name is derived from the chunk-config hash
(R0.2), so changing chunk size / overlap / embedding model never reuses old vectors.
Chroma holds only ids + vectors + filterable metadata; text lives in SQLite.
"""

from __future__ import annotations

import threading
from pathlib import Path

import chromadb

from src import config

_clients: dict[str, chromadb.api.ClientAPI] = {}
_lock = threading.Lock()  # chromadb's client registry is not thread-safe: two requests racing to create
#                          the first PersistentClient crash with 'Could not connect to tenant default_tenant'


def _client(path=None):
    path = str(path or config.CHROMA_DIR)
    with _lock:
        if path not in _clients:
            Path(path).mkdir(parents=True, exist_ok=True)
            _clients[path] = chromadb.PersistentClient(path=path)
        return _clients[path]


def reset_clients() -> None:
    with _lock:
        _clients.clear()


def get_collection(cfg: config.ChunkConfig, path=None):
    return _client(path).get_or_create_collection(name=cfg.collection, metadata={"hnsw:space": "cosine"})


def replace_transcript_vectors(cfg, transcript_id: int, ids, embeddings, metadatas, path=None) -> None:
    col = get_collection(cfg, path)
    col.delete(where={"transcript_id": transcript_id})  # refresh, never skip (fixes F4)
    if ids:
        col.add(ids=list(ids), embeddings=[e.tolist() for e in embeddings], metadatas=list(metadatas))


def _where(filters: dict | None):
    if not filters:
        return None
    conds = [{k: {"$eq": v}} for k, v in filters.items() if v not in (None, "") and k in
             {"ticker", "quarter", "year", "section", "role"}]
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def search(cfg, query_embedding, k: int = 50, filters: dict | None = None, path=None) -> list[tuple[str, float]]:
    col = get_collection(cfg, path)
    n = col.count()
    if n == 0:
        return []
    kwargs = {"query_embeddings": [query_embedding.tolist()], "n_results": min(k, n)}
    w = _where(filters)
    if w:
        kwargs["where"] = w
    res = col.query(**kwargs)
    return [(cid, round(1.0 - d, 4)) for cid, d in zip(res["ids"][0], res["distances"][0])]


def count(cfg, path=None) -> int:
    return get_collection(cfg, path).count()
