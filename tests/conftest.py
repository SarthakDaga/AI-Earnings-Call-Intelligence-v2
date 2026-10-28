"""Shared fixtures: an isolated SQLite DB + Chroma dir built from tiny invented transcripts."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config  # noqa: E402
from src.database import queries, schema  # noqa: E402
from src.ingestion.loader import load_all_transcripts  # noqa: E402
from src.ingestion.turns import parse_transcript, to_clean_text  # noqa: E402
from src.retrieval import dense  # noqa: E402
from src.retrieval.embedder import FakeEmbedder  # noqa: E402
from src.retrieval.index import build_index  # noqa: E402
from src.retrieval.reranker import FakeReranker  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "transcripts"


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(schema, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    dense.reset_clients()
    schema.init_db()
    yield tmp_path
    dense.reset_clients()


@pytest.fixture
def loaded(tmp_db):
    """DB populated with the two fixture transcripts (no indexes yet)."""
    for t in load_all_transcripts(FIXTURE_DIR, manifest_path=tmp_db / "none.csv"):
        parsed = parse_transcript(t["raw_text"])
        queries.insert_transcript(t["company"], t["ticker"], t["quarter"], t["year"],
                                  t["raw_text"], to_clean_text(parsed), date=t["date"])
    return tmp_db


@pytest.fixture
def retriever(loaded):
    cfg = config.get_chunk_config("v2")
    emb = FakeEmbedder()
    build_index(cfg, emb, chroma_path=loaded / "chroma", progress=lambda *_: None)
    return Retriever(cfg, embedder=emb, reranker=FakeReranker(), chroma_path=loaded / "chroma").ensure_ready()
