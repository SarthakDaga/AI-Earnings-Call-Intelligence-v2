"""
scripts/check_db.py

Quick sanity check after run_pipeline.py: what is in SQLite, per chunk config.

    python scripts/check_db.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHUNK_CONFIGS  # noqa: E402
from src.database import queries  # noqa: E402
from src.database.schema import get_connection, init_db  # noqa: E402


def main() -> None:
    init_db()
    conn = get_connection()
    n_t = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
    n_s = conn.execute("SELECT COUNT(*) FROM sentiment_results").fetchone()[0]
    n_sum = conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
    conn.close()
    print(f"transcripts: {n_t}   sentiment rows: {n_s}   summaries: {n_sum}")
    for name, cfg in CHUNK_CONFIGS.items():
        st = queries.corpus_stats(cfg.hash)
        if st["total_chunks"]:
            print(f"config {name:6s} ({cfg.hash}): {st['total_chunks']:6d} chunks, "
                  f"{st['transcripts']} transcripts, tickers={st['companies']}")
        else:
            print(f"config {name:6s} ({cfg.hash}): no chunks (not built)")
    if n_t != n_s:
        print(f"\nNOTE: {n_t - n_s} transcript(s) have no sentiment yet - re-run run_pipeline.py")


if __name__ == "__main__":
    main()