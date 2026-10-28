"""
src/database/queries.py
All database reads/writes. Inserts are upserts, so re-running is idempotent (R0.1).
"""

import json
from typing import Optional

from .schema import get_connection


# ── TRANSCRIPTS ──────────────────────────────────────────────
def insert_transcript(company: str, ticker: str, quarter: str, year: int,
                      raw_text: str, processed_text: str,
                      date: str = None, source: str = None,
                      source_url: str = None) -> int:
    """Upsert on (ticker, quarter, year). The row id is stable across re-runs."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO transcripts
            (company, ticker, quarter, year, date, source, source_url, raw_text, processed_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, quarter, year) DO UPDATE SET
            company=excluded.company,
            date=COALESCE(excluded.date, transcripts.date),
            source=excluded.source,
            source_url=COALESCE(excluded.source_url, transcripts.source_url),
            raw_text=excluded.raw_text,
            processed_text=excluded.processed_text
    """, (company, ticker, quarter, year, date, source, source_url, raw_text, processed_text))
    conn.commit()
    row = conn.execute("SELECT id FROM transcripts WHERE ticker=? AND quarter=? AND year=?",
                       (ticker, quarter, year)).fetchone()
    conn.close()
    return row["id"]


def get_all_transcripts() -> list[dict]:
    conn = get_connection()
    rows = [dict(r) for r in conn.execute("""
        SELECT id, company, ticker, quarter, year, date, source, source_url, created_at
        FROM transcripts ORDER BY ticker, year DESC, quarter DESC
    """).fetchall()]
    conn.close()
    return rows


def get_transcript_by_id(transcript_id: int) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM transcripts WHERE id = ?", (transcript_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_transcript_by_key(ticker: str, quarter: str, year: int) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM transcripts WHERE ticker=? AND quarter=? AND year=?",
                       (ticker, quarter, year)).fetchone()
    conn.close()
    return dict(row) if row else None


def transcript_exists(company: str, ticker: str, quarter: str, year: int) -> Optional[int]:
    conn = get_connection()
    row = conn.execute("SELECT id FROM transcripts WHERE ticker=? AND quarter=? AND year=?",
                       (ticker, quarter, year)).fetchone()
    conn.close()
    return row["id"] if row else None


def delete_transcript_artifacts(transcript_id: int) -> None:
    """Used by --force: drop derived rows so they are rebuilt, not duplicated."""
    conn = get_connection()
    conn.execute("DELETE FROM sentiment_results WHERE transcript_id=?", (transcript_id,))
    conn.execute("DELETE FROM summaries WHERE transcript_id=?", (transcript_id,))
    conn.execute("DELETE FROM chunks WHERE transcript_id=?", (transcript_id,))
    conn.commit()
    conn.close()


# ── SENTIMENT ────────────────────────────────────────────────
def insert_sentiment(transcript_id: int, positive_pct: float, neutral_pct: float,
                     negative_pct: float, net_score: float,
                     top_positive: list, top_negative: list, sentence_scores: list):
    conn = get_connection()
    conn.execute("""
        INSERT INTO sentiment_results
            (transcript_id, positive_pct, neutral_pct, negative_pct, net_score,
             top_positive_sentences, top_negative_sentences, sentence_scores)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(transcript_id) DO UPDATE SET
            positive_pct=excluded.positive_pct, neutral_pct=excluded.neutral_pct,
            negative_pct=excluded.negative_pct, net_score=excluded.net_score,
            top_positive_sentences=excluded.top_positive_sentences,
            top_negative_sentences=excluded.top_negative_sentences,
            sentence_scores=excluded.sentence_scores,
            computed_at=CURRENT_TIMESTAMP
    """, (transcript_id, positive_pct, neutral_pct, negative_pct, net_score,
          json.dumps(top_positive), json.dumps(top_negative), json.dumps(sentence_scores)))
    conn.commit()
    conn.close()


def get_sentiment(transcript_id: int) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM sentiment_results WHERE transcript_id = ?",
                       (transcript_id,)).fetchone()
    conn.close()
    if not row:
        return None
    result = dict(row)
    for f in ("top_positive_sentences", "top_negative_sentences", "sentence_scores"):
        result[f] = json.loads(result[f] or "[]")
    return result


# ── SUMMARIES ────────────────────────────────────────────────
def insert_summary(transcript_id: int, executive_summary: str, key_takeaways: list,
                   risks: list, opportunities: list, notable_quotes: list,
                   analyst_signal: str, signal_justification: str, model_used: str):
    conn = get_connection()
    conn.execute("""
        INSERT INTO summaries
            (transcript_id, executive_summary, key_takeaways, risks, opportunities,
             notable_quotes, analyst_signal, signal_justification, model_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(transcript_id) DO UPDATE SET
            executive_summary=excluded.executive_summary, key_takeaways=excluded.key_takeaways,
            risks=excluded.risks, opportunities=excluded.opportunities,
            notable_quotes=excluded.notable_quotes, analyst_signal=excluded.analyst_signal,
            signal_justification=excluded.signal_justification, model_used=excluded.model_used,
            computed_at=CURRENT_TIMESTAMP
    """, (transcript_id, executive_summary, json.dumps(key_takeaways), json.dumps(risks),
          json.dumps(opportunities), json.dumps(notable_quotes),
          analyst_signal, signal_justification, model_used))
    conn.commit()
    conn.close()


def get_summary(transcript_id: int) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM summaries WHERE transcript_id = ?",
                       (transcript_id,)).fetchone()
    conn.close()
    if not row:
        return None
    result = dict(row)
    for f in ("key_takeaways", "risks", "opportunities", "notable_quotes"):
        result[f] = json.loads(result[f] or "[]")
    return result


def delete_summary(transcript_id: int) -> None:
    conn = get_connection()
    conn.execute("DELETE FROM summaries WHERE transcript_id=?", (transcript_id,))
    conn.commit()
    conn.close()


# ── CHUNKS ───────────────────────────────────────────────────
def replace_chunks(config_hash: str, transcript_id: int, rows: list[dict]) -> None:
    """Delete then insert, so re-chunking always refreshes (fixes F4)."""
    conn = get_connection()
    conn.execute("DELETE FROM chunks WHERE config_hash=? AND transcript_id=?",
                 (config_hash, transcript_id))
    conn.executemany("""
        INSERT INTO chunks (chunk_id, config_hash, transcript_id, ticker, quarter, year,
                            section, speaker, role, chunk_index, text)
        VALUES (:chunk_id, :config_hash, :transcript_id, :ticker, :quarter, :year,
                :section, :speaker, :role, :chunk_index, :text)
    """, [{**r, "config_hash": config_hash, "transcript_id": transcript_id} for r in rows])
    conn.commit()
    conn.close()


def get_chunks(config_hash: str) -> list[dict]:
    conn = get_connection()
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM chunks WHERE config_hash=? ORDER BY transcript_id, chunk_index",
        (config_hash,)).fetchall()]
    conn.close()
    return rows


def chunks_exist(config_hash: str, transcript_id: int) -> bool:
    conn = get_connection()
    row = conn.execute("SELECT 1 FROM chunks WHERE config_hash=? AND transcript_id=? LIMIT 1",
                       (config_hash, transcript_id)).fetchone()
    conn.close()
    return row is not None


def corpus_stats(config_hash: str) -> dict:
    """Exact stats over all chunks (v1 sampled only 1,000 — fixes F9)."""
    conn = get_connection()
    n = conn.execute("SELECT COUNT(*) c FROM chunks WHERE config_hash=?", (config_hash,)).fetchone()["c"]
    tickers = [r["ticker"] for r in conn.execute(
        "SELECT DISTINCT ticker FROM chunks WHERE config_hash=? ORDER BY ticker", (config_hash,))]
    t = conn.execute("SELECT COUNT(DISTINCT transcript_id) c FROM chunks WHERE config_hash=?",
                     (config_hash,)).fetchone()["c"]
    conn.close()
    return {"total_chunks": n, "companies": tickers, "transcripts": t}


# ── PRICE CACHE ──────────────────────────────────────────────
def upsert_prices(ticker: str, rows: list[tuple[str, float]]) -> None:
    conn = get_connection()
    conn.executemany("INSERT OR REPLACE INTO price_cache (ticker, date, close) VALUES (?, ?, ?)",
                     [(ticker, d, c) for d, c in rows])
    conn.commit()
    conn.close()


def get_prices(ticker: str, start: str, end: str) -> list[tuple[str, float]]:
    conn = get_connection()
    rows = conn.execute("""SELECT date, close FROM price_cache
                           WHERE ticker=? AND date>=? AND date<=? ORDER BY date""",
                        (ticker, start, end)).fetchall()
    conn.close()
    return [(r["date"], r["close"]) for r in rows]


def log_price_fetch(ticker: str, start: str, end: str) -> None:
    conn = get_connection()
    conn.execute("INSERT OR REPLACE INTO price_fetch_log (ticker, start, end) VALUES (?, ?, ?)",
                 (ticker, start, end))
    conn.commit()
    conn.close()


def price_fetch_logged(ticker: str, start: str, end: str) -> bool:
    conn = get_connection()
    row = conn.execute("SELECT 1 FROM price_fetch_log WHERE ticker=? AND start=? AND end=?",
                       (ticker, start, end)).fetchone()
    conn.close()
    return row is not None
