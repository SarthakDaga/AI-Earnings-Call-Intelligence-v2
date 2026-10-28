import importlib
import shutil

from src import config
from src.database import queries
from src.ingestion import loader
from src.retrieval.embedder import FakeEmbedder

FIX = "tests/fixtures/transcripts"


def _fake_scorer(sentences):
    return [{"text": s, "label": "positive", "score": .9, "positive": .9, "negative": .05, "neutral": .05}
            for s in sentences]


def test_pipeline_end_to_end_idempotent_and_force(tmp_db, monkeypatch):
    tdir = tmp_db / "transcripts"
    shutil.copytree(FIX, tdir)
    (tmp_db / "manifest.csv").write_text("ticker,fiscal_quarter,fiscal_year,call_date,source_url\n"
                                         "ACME,Q1,2025,2025-04-30,https://example.com/a\n")
    monkeypatch.setattr(loader, "TRANSCRIPTS_DIR", tdir)
    monkeypatch.setattr(loader, "MANIFEST_PATH", tmp_db / "manifest.csv")
    monkeypatch.setattr("src.sentiment.finbert.score_sentences", _fake_scorer)
    import src.retrieval.embedder as emb
    monkeypatch.setattr(emb, "Embedder", lambda name: FakeEmbedder())
    from src.retrieval import dense
    monkeypatch.setattr(dense, "_client", lambda path=None: _client)
    import chromadb
    _client = chromadb.PersistentClient(path=str(tmp_db / "chroma"))
    run_pipeline = importlib.import_module("scripts.run_pipeline").run_pipeline

    run_pipeline(configs=["v1", "v2"])
    cfg1, cfg2 = config.get_chunk_config("v1"), config.get_chunk_config("v2")
    n1, n2 = len(queries.get_chunks(cfg1.hash)), len(queries.get_chunks(cfg2.hash))
    assert n1 > 0 and n2 > 0 and len(queries.get_all_transcripts()) == 2
    t = queries.get_transcript_by_key("ACME", "Q1", 2025)
    assert t["date"] == "2025-04-30" and t["source_url"] == "https://example.com/a"
    s = queries.get_sentiment(t["id"])
    assert {x["section"] for x in s["sentence_scores"]} == {"prepared", "qa"}

    run_pipeline(configs=["v1", "v2"])                       # idempotent
    assert len(queries.get_chunks(cfg2.hash)) == n2
    run_pipeline(force=True, configs=["v1", "v2"])           # force replaces, never duplicates (F4)
    assert len(queries.get_chunks(cfg1.hash)) == n1 and len(queries.get_chunks(cfg2.hash)) == n2
    assert dense.get_collection(cfg2).count() == n2
    assert len(queries.get_all_transcripts()) == 2
