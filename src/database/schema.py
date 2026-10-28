"""
src/database/schema.py
SQLite schema (v2): UNIQUE constraints + WAL (R0.1), chunk store for BM25,
and a price cache for the yfinance tool.
"""

import sqlite3
from pathlib import Path

from src import config

DB_PATH: Path = config.DB_PATH
SCHEMA_VERSION = 2


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _migrate_if_old(conn: sqlite3.Connection) -> None:
    """v1 databases had no UNIQUE constraints (and could hold duplicates).
    Back the old file up and start fresh rather than guess at a merge."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    has_tables = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='transcripts'"
    ).fetchone()
    if has_tables and version < SCHEMA_VERSION:
        conn.close()
        backup = DB_PATH.with_suffix(".v1.bak")
        DB_PATH.replace(backup)
        print(f"[DB] Old v1 schema found. Backed up to {backup.name}; creating v2 schema. "
              f"Re-run scripts/run_pipeline.py to repopulate.")
        return None
    return None


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        probe = sqlite3.connect(str(DB_PATH))
        _migrate_if_old(probe)
        try:
            probe.close()
        except Exception:
            pass

    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS transcripts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            company         TEXT    NOT NULL,
            ticker          TEXT    NOT NULL,
            quarter         TEXT    NOT NULL,
            year            INTEGER NOT NULL,
            date            TEXT,            -- call date from manifest.csv
            source          TEXT,
            source_url      TEXT,
            raw_text        TEXT    NOT NULL,
            processed_text  TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (ticker, quarter, year)
        );

        CREATE TABLE IF NOT EXISTS sentiment_results (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript_id           INTEGER NOT NULL UNIQUE REFERENCES transcripts(id) ON DELETE CASCADE,
            positive_pct            REAL,
            neutral_pct             REAL,
            negative_pct            REAL,
            net_score               REAL,
            top_positive_sentences  TEXT,
            top_negative_sentences  TEXT,
            sentence_scores         TEXT,   -- JSON: [{text,label,score,positive,negative,neutral,section}]
            computed_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS summaries (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript_id        INTEGER NOT NULL UNIQUE REFERENCES transcripts(id) ON DELETE CASCADE,
            executive_summary    TEXT,
            key_takeaways        TEXT,
            risks                TEXT,
            opportunities        TEXT,
            notable_quotes       TEXT,
            analyst_signal       TEXT,   -- displayed as "Tone signal"
            signal_justification TEXT,
            model_used           TEXT,
            computed_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Chunk store: the source of truth for BM25 and for resolving dense hits.
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id       TEXT NOT NULL,
            config_hash    TEXT NOT NULL,
            transcript_id  INTEGER NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
            ticker         TEXT NOT NULL,
            quarter        TEXT NOT NULL,
            year           INTEGER NOT NULL,
            section        TEXT,
            speaker        TEXT,
            role           TEXT,
            chunk_index    INTEGER,
            text           TEXT NOT NULL,
            PRIMARY KEY (config_hash, chunk_id)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_tid ON chunks(config_hash, transcript_id);

        CREATE TABLE IF NOT EXISTS price_cache (
            ticker  TEXT NOT NULL,
            date    TEXT NOT NULL,
            close   REAL NOT NULL,
            PRIMARY KEY (ticker, date)
        );
        CREATE TABLE IF NOT EXISTS price_fetch_log (
            ticker TEXT NOT NULL, start TEXT NOT NULL, end TEXT NOT NULL,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, start, end)
        );
    """)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"[DB] Initialized at {DB_PATH}")
