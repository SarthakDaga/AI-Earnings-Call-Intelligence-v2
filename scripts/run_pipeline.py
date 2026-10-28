"""
scripts/run_pipeline.py

Offline processing: ingest -> structure-aware parse -> FinBERT sentiment ->
chunk + embed for one or more retrieval configs. Idempotent: re-running skips
finished work; --force rebuilds (and replaces, never duplicates).

    python scripts/run_pipeline.py                      # v2 index (default)
    python scripts/run_pipeline.py --configs v1 v2      # also build the A0 baseline index
    python scripts/run_pipeline.py --force
    python scripts/run_pipeline.py --skip-sentiment     # retrieval only (no torch/FinBERT needed)

Summaries are generated on demand (dashboard/API) because they need an LLM.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.database import queries  # noqa: E402
from src.database.schema import init_db  # noqa: E402
from src.ingestion.loader import load_all_transcripts  # noqa: E402
from src.ingestion.turns import parse_transcript, to_clean_text, validate_header  # noqa: E402
from src.sentiment.pipeline import sentences_with_sections  # noqa: E402


def run_pipeline(force: bool = False, configs: list[str] | None = None,
                 skip_sentiment: bool = False, skip_embeddings: bool = False) -> None:
    configs = configs or ["v2"]
    print("=" * 60 + "\n  Earnings Call Intelligence - pipeline\n" + "=" * 60)
    init_db()
    raw = load_all_transcripts()
    if not raw:
        print("\nNo transcripts found in data/transcripts/ (files named TICKER_Q#_YYYY.txt).")
        return

    ids = []
    for t in raw:
        label = f"{t['ticker']} {t['quarter']} {t['year']}"
        print(f"\n-- {label}")
        parsed = parse_transcript(t["raw_text"], collapse_company=True)
        for w in parsed.warnings:
            print(f"   warning: {w}")
        for p in validate_header(parsed.header, fiscal_quarter=t["quarter"], fiscal_year=t["year"], call_date=t["date"]):
            print(f"   HEADER MISMATCH: {p}")
        if not t["date"]:
            print("   note: no call_date in data/manifest.csv (price-reaction tool needs it)")
        existing = queries.transcript_exists(t["company"], t["ticker"], t["quarter"], t["year"])
        if existing and force:
            queries.delete_transcript_artifacts(existing)
        tid = queries.insert_transcript(t["company"], t["ticker"], t["quarter"], t["year"], t["raw_text"],
                                        to_clean_text(parsed), date=t["date"], source=t["source"],
                                        source_url=t.get("source_url"))
        ids.append(tid)
        print(f"   stored (id={tid}), {len(parsed.turns)} turns, qa={'yes' if parsed.qa_found else 'NO'}")

        if not skip_sentiment:
            have = queries.get_sentiment(tid)
            if have and not force:
                print(f"   sentiment cached (net {have['net_score']:+.3f})")
            else:
                from src.sentiment.finbert import run_sentiment_pipeline
                sentences, sections = sentences_with_sections(parsed)
                agg, scores = run_sentiment_pipeline(sentences, sections)
                queries.insert_sentiment(tid, agg["positive_pct"], agg["neutral_pct"], agg["negative_pct"],
                                         agg["net_score"], agg["top_positive_sentences"],
                                         agg["top_negative_sentences"], scores)
                print(f"   sentiment: pos {agg['positive_pct']}% / neg {agg['negative_pct']}% / net {agg['net_score']:+.3f}")

    if not skip_embeddings:
        from src.retrieval.embedder import Embedder
        from src.retrieval.index import build_index
        transcripts = [queries.get_transcript_by_id(i) for i in ids]
        for name in configs:
            cfg = config.get_chunk_config(name)
            print(f"\n== indexing config '{cfg.name}' -> collection {cfg.collection}")
            build_index(cfg, Embedder(cfg.embed_model), transcripts, force=force)
    print("\nDone. Next:\n  uvicorn src.api.main:app --port 7860\n  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the earnings call processing pipeline")
    ap.add_argument("--force", action="store_true", help="rebuild everything, replacing old rows/vectors")
    ap.add_argument("--configs", nargs="+", default=["v2"], choices=list(config.CHUNK_CONFIGS))
    ap.add_argument("--skip-sentiment", action="store_true")
    ap.add_argument("--skip-embeddings", action="store_true")
    a = ap.parse_args()
    run_pipeline(a.force, a.configs, a.skip_sentiment, a.skip_embeddings)
