"""
scripts/check_truncation.py

Quantifies finding F1: what fraction of chunks exceed the embedding model's max
input length and are silently truncated? Run it for each config.

    python scripts/check_truncation.py --configs v1 v2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.database import queries  # noqa: E402
from src.preprocessing.chunker import Chunk  # noqa: E402
from src.retrieval.embedder import Embedder  # noqa: E402
from src.retrieval.index import embed_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["v1", "v2"])
    for name in ap.parse_args().configs:
        cfg = config.get_chunk_config(name)
        rows = queries.get_chunks(cfg.hash)
        if not rows:
            print(f"{name}: no chunks indexed")
            continue
        emb = Embedder(cfg.embed_model)
        tr = {t["id"]: t for t in (queries.get_transcript_by_id(i["id"]) for i in queries.get_all_transcripts())}
        texts = [embed_text(cfg, tr[r["transcript_id"]], Chunk(r["text"], r["speaker"] or "", r["role"] or "",
                                                               r["section"] or "")) for r in rows]
        counts = emb.token_counts(texts)
        limit = emb.max_seq_length
        over = [c for c in counts if c > limit]
        print(f"{name:6s} model={cfg.embed_model} max_seq={limit} chunks={len(counts)} "
              f"median_tokens={sorted(counts)[len(counts) // 2]} truncated={len(over)} ({len(over) / len(counts):.0%})")


if __name__ == "__main__":
    main()
