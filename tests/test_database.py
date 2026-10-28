"""SQLite operations on an isolated temp database (see conftest.tmp_db)."""

import sqlite3

import pytest

from src.database import queries, schema


def _t(ticker="NVDA", q="Q1", year=2024, **kw):
    return queries.insert_transcript("NVIDIA", ticker, q, year, kw.get("raw", "raw"), kw.get("clean", "clean"),
                                     date=kw.get("date"))


def test_init_creates_tables(tmp_db):
    conn = schema.get_connection()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert {"transcripts", "sentiment_results", "summaries", "chunks", "price_cache"} <= tables


def test_wal_enabled(tmp_db):
    conn = schema.get_connection()
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    conn.close()


def test_insert_is_upsert_with_stable_id(tmp_db):
    a = _t(raw="v1 text")
    b = _t(raw="v2 text", date="2024-02-21")
    assert a == b
    row = queries.get_transcript_by_id(a)
    assert row["raw_text"] == "v2 text" and row["date"] == "2024-02-21"
    assert len(queries.get_all_transcripts()) == 1


def test_unique_constraint_enforced_in_schema(tmp_db):
    _t()
    conn = schema.get_connection()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO transcripts (company,ticker,quarter,year,raw_text) VALUES ('x','NVDA','Q1',2024,'t')")
    conn.close()


def test_transcript_exists(tmp_db):
    _t("AAPL", "Q2", 2024)
    assert queries.transcript_exists("Apple", "AAPL", "Q2", 2024)
    assert queries.transcript_exists("Apple", "AAPL", "Q3", 2024) is None


def test_sentiment_upsert_no_duplicates(tmp_db):
    tid = _t()
    for pct in (48.0, 50.0):
        queries.insert_sentiment(tid, pct, 39.0, 13.0, 0.35, [{"text": "Great", "score": .97}],
                                 [{"text": "Margin pressure", "score": .84}], [{"text": "T", "label": "positive"}])
    s = queries.get_sentiment(tid)
    assert s["positive_pct"] == 50.0 and s["top_positive_sentences"][0]["text"] == "Great"
    conn = schema.get_connection()
    assert conn.execute("SELECT COUNT(*) FROM sentiment_results").fetchone()[0] == 1
    conn.close()


def test_summary_upsert_and_delete(tmp_db):
    tid = _t("AAPL", "Q2", 2024)
    for sig in ("Neutral", "Bullish"):
        queries.insert_summary(tid, "Strong quarter.", ["Services grew 14%"], ["FX"], ["AI"],
                               ["a verbatim quote of five words"], sig, "why", "llama3.2:3b")
    s = queries.get_summary(tid)
    assert s["analyst_signal"] == "Bullish" and "Services grew 14%" in s["key_takeaways"]
    queries.delete_summary(tid)
    assert queries.get_summary(tid) is None


def test_force_cleanup_removes_derived_rows(tmp_db):
    tid = _t()
    queries.insert_sentiment(tid, 1, 1, 1, 0, [], [], [])
    queries.replace_chunks("h", tid, [{"chunk_id": "c0", "ticker": "NVDA", "quarter": "Q1", "year": 2024,
                                       "section": "qa", "speaker": "A", "role": "analyst",
                                       "chunk_index": 0, "text": "hello"}])
    queries.delete_transcript_artifacts(tid)
    assert queries.get_sentiment(tid) is None and not queries.chunks_exist("h", tid)


def test_corpus_stats_exact(tmp_db):
    tid = _t()
    rows = [{"chunk_id": f"c{i}", "ticker": "NVDA", "quarter": "Q1", "year": 2024, "section": "",
             "speaker": "", "role": "", "chunk_index": i, "text": "t"} for i in range(1500)]
    queries.replace_chunks("h", tid, rows)
    assert queries.corpus_stats("h") == {"total_chunks": 1500, "companies": ["NVDA"], "transcripts": 1}


def test_old_v1_database_is_backed_up(tmp_path, monkeypatch):
    p = tmp_path / "old.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE transcripts (id INTEGER PRIMARY KEY, company TEXT)")
    c.commit()
    c.close()
    monkeypatch.setattr(schema, "DB_PATH", p)
    schema.init_db()
    assert p.with_suffix(".v1.bak").exists()
    conn = schema.get_connection()
    assert conn.execute("PRAGMA user_version").fetchone()[0] == schema.SCHEMA_VERSION
    conn.close()


def test_missing_returns_none(tmp_db):
    assert queries.get_transcript_by_id(9999) is None
    assert queries.get_sentiment(9999) is None
    assert queries.get_summary(9999) is None
