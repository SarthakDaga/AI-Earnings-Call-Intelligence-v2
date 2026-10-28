import pytest

from src.retrieval.bm25 import BM25Index, tokenize
from src.retrieval.fusion import rrf
from src.retrieval.retriever import MODES


def test_finance_tokenizer_keeps_tokens():
    toks = tokenize("Q3 guidance of $4.8 billion, FY25 margin 51% and end-to-end supply")
    for t in ("q3", "$4.8", "fy25", "51%", "end-to-end"):
        assert t in toks
    assert "the" not in tokenize("the margin")


def test_tokenizer_matches_normalized_numbers():
    assert tokenize("$68,000,000,000") == tokenize("$68 billion")


def test_rrf_prefers_agreement():
    fused = rrf([["a", "b", "c"], ["c", "a", "d"]], k=60)
    assert [d for d, _ in fused][:2] == ["a", "c"]
    assert dict(fused)["a"] > dict(fused)["b"]


def test_bm25_filter_and_scores():
    chunks = [{"chunk_id": "1", "text": "servo motor supply tight", "ticker": "A"},
              {"chunk_id": "2", "text": "servo motor supply improving", "ticker": "B"},
              {"chunk_id": "3", "text": "unrelated pricing talk", "ticker": "A"}]
    idx = BM25Index(chunks)
    assert {c for c, _ in idx.search("servo supply")} == {"1", "2"}
    assert [c for c, _ in idx.search("servo supply", allow=lambda c: c["ticker"] == "A")] == ["1"]


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_returns_relevant_hit(retriever, mode):
    hits = retriever.search("servo motor supply constraints", k=3, mode=mode)
    assert hits and any("servo" in h["text"].lower() for h in hits)
    assert all({"chunk_id", "text", "ticker", "quarter", "year", "score"} <= set(h) for h in hits)


def test_filters_apply_to_both_legs(retriever):
    for mode in ("dense", "bm25", "hybrid"):
        hits = retriever.search("revenue guidance", k=5, mode=mode, filters={"quarter": "Q2", "ticker": "acme"})
        assert hits and all(h["quarter"] == "Q2" for h in hits)
    qa = retriever.search("pricing", k=5, mode="hybrid", filters={"section": "qa"})
    assert qa and all(h["section"] == "qa" for h in qa)


def test_bad_mode_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="nope")


def test_reindex_refreshes_instead_of_skipping(loaded):
    """F4 regression: rebuilding with force must replace vectors, not keep stale ones."""
    from src import config
    from src.database import queries
    from src.retrieval.embedder import FakeEmbedder
    from src.retrieval.index import build_index
    cfg = config.get_chunk_config("v2")
    kw = dict(chroma_path=loaded / "chroma", progress=lambda *_: None)
    build_index(cfg, FakeEmbedder(), **kw)
    n1 = len(queries.get_chunks(cfg.hash))
    r = build_index(cfg, FakeEmbedder(), force=True, **kw)
    assert r["built"] == 2 and len(queries.get_chunks(cfg.hash)) == n1  # no duplicates


def test_collection_name_versioned_by_config():
    from src import config
    assert config.get_chunk_config("v1").collection != config.get_chunk_config("v2").collection
    assert config.get_chunk_config("v2").hash != config.get_chunk_config("v2bge").hash


# ── regressions from the first real run ──────────────────────
def test_chroma_client_creation_is_thread_safe(tmp_path):
    """Concurrent first requests used to crash chromadb ('Could not connect to tenant default_tenant')."""
    import threading

    from src.retrieval import dense
    for _ in range(3):
        dense.reset_clients()
        path = tmp_path / f"c{_}"
        errors = []

        def go(p=path, errs=errors):
            try:
                dense._client(p)
            except Exception as e:  # noqa: BLE001
                errs.append(repr(e))

        threads = [threading.Thread(target=go) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert not errors, errors
        assert len({id(dense._client(path)) for _ in range(3)}) == 1  # one shared client
    dense.reset_clients()


def test_models_load_once_under_concurrency(monkeypatch):
    """Seven simultaneous requests used to load the embedding model seven times."""
    import sys
    import threading
    import time
    import types

    loads = []

    class Slow:
        def __init__(self, name):
            loads.append(name)
            time.sleep(0.2)

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Slow))
    from src.retrieval.embedder import Embedder
    emb = Embedder("any-model")
    threads = [threading.Thread(target=emb._load) for _ in range(7)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(loads) == 1


def test_rerank_falls_back_to_hybrid_when_reranker_breaks(retriever):
    class Broken:
        loaded = False

        def score(self, q, texts):
            raise OSError("model files missing")

    expected = [h["chunk_id"] for h in retriever.search("revenue", k=3, mode="hybrid")]
    retriever._reranker = Broken()
    hits = retriever.search("revenue", k=3, mode="hybrid_rerank")
    assert [h["chunk_id"] for h in hits] == expected
    assert "model files missing" in retriever.status()["reranker_error"]


def test_rerank_does_not_wait_for_a_reranker_that_is_still_loading(retriever):
    class Loading:
        loaded, loading = False, True

        def score(self, q, texts):
            raise AssertionError("must not block on a model that is still loading")

    retriever._reranker = Loading()
    assert retriever.search("revenue", k=3, mode="hybrid_rerank")
