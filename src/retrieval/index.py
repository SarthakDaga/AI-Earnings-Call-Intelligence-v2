"""
src/retrieval/index.py
Builds chunks + vectors for a ChunkConfig. v1 reproduces the original behaviour
(v1 cleaner + 400-word windows) so A0 stays a faithful baseline.
"""

from __future__ import annotations

from src import config
from src.database import queries
from src.ingestion.parser import clean_transcript
from src.ingestion.turns import parse_transcript
from src.preprocessing.chunker import Chunk, chunk_turns, context_header
from src.preprocessing.tokenizer import chunk_text
from src.retrieval import dense


def make_chunks(cfg: config.ChunkConfig, t: dict) -> list[Chunk]:
    if cfg.parser == "v1":
        pieces = chunk_text(clean_transcript(t["raw_text"]), cfg.chunk_size, cfg.overlap)
        return [Chunk(p, "", "", "", i) for i, p in enumerate(pieces)]
    parsed = parse_transcript(t["raw_text"], normalize=True, collapse_company=True)
    return chunk_turns(parsed, cfg.chunk_size, cfg.overlap)


def embed_text(cfg: config.ChunkConfig, t: dict, c: Chunk) -> str:
    if cfg.context_header:
        return context_header(t["ticker"], t["quarter"], t["year"], c) + "\n" + c.text
    return c.text


def build_index(cfg: config.ChunkConfig, embedder, transcripts: list[dict] | None = None,
                force: bool = False, chroma_path=None, progress=print) -> dict:
    """Index every transcript missing from this config (or all, with force)."""
    transcripts = transcripts if transcripts is not None else [
        queries.get_transcript_by_id(t["id"]) for t in queries.get_all_transcripts()]
    built = skipped = 0
    for t in transcripts:
        tid = t["id"]
        have_vectors = False
        if queries.chunks_exist(cfg.hash, tid) and not force:
            have_vectors = True
        if have_vectors:
            skipped += 1
            progress(f"  ✓ {t['ticker']} {t['quarter']} {t['year']} already indexed ({cfg.name})")
            continue
        chunks = make_chunks(cfg, t)
        if not chunks:
            progress(f"  ! {t['ticker']} {t['quarter']} {t['year']}: no chunks produced")
            continue
        vecs = embedder.encode_docs([embed_text(cfg, t, c) for c in chunks])
        ids = [f"{t['ticker']}_{t['quarter']}_{t['year']}_{cfg.name}_{c.index}" for c in chunks]
        rows = [{"chunk_id": cid, "ticker": t["ticker"], "quarter": t["quarter"], "year": t["year"],
                 "section": c.section, "speaker": c.speaker, "role": c.role,
                 "chunk_index": c.index, "text": c.text} for cid, c in zip(ids, chunks)]
        queries.replace_chunks(cfg.hash, tid, rows)
        metas = [{"transcript_id": tid, "ticker": t["ticker"], "quarter": t["quarter"],
                  "year": t["year"], "section": c.section or "", "role": c.role or ""} for c in chunks]
        dense.replace_transcript_vectors(cfg, tid, ids, vecs, metas, chroma_path)
        built += 1
        progress(f"  ✓ {t['ticker']} {t['quarter']} {t['year']}: {len(chunks)} chunks ({cfg.name})")
    return {"built": built, "skipped": skipped, "config": cfg.name, "collection": cfg.collection}
