"""
eval/smoke.py   -   retrieval regression gate for CI (S5).

Builds an index over the tiny INVENTED fixture transcripts with a deterministic
hashing embedder (no model downloads), runs the retriever in every mode, and
fails if hybrid Recall@5 falls below the committed baseline minus a tolerance.

    python -m eval.smoke            # check against eval/baseline_smoke.json
    python -m eval.smoke --update   # rewrite the baseline after an intentional change
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASELINE = ROOT / "eval" / "baseline_smoke.json"
TOLERANCE = 0.05
GATE_MODE = "hybrid"


def run() -> dict:
    from eval import metrics
    from eval.run_eval import eval_retrieval, summarize_retrieval
    from src import config
    from src.database import queries, schema
    from src.ingestion.loader import load_all_transcripts
    from src.ingestion.turns import parse_transcript, to_clean_text
    from src.retrieval import dense
    from src.retrieval.embedder import FakeEmbedder
    from src.retrieval.index import build_index
    from src.retrieval.reranker import FakeReranker
    from src.retrieval.retriever import Retriever

    tmp = Path(tempfile.mkdtemp())
    schema.DB_PATH = tmp / "smoke.db"
    dense.reset_clients()
    schema.init_db()
    for t in load_all_transcripts(ROOT / "tests/fixtures/transcripts", manifest_path=tmp / "none.csv"):
        queries.insert_transcript(t["company"], t["ticker"], t["quarter"], t["year"], t["raw_text"],
                                  to_clean_text(parse_transcript(t["raw_text"])))
    cfg, emb = config.get_chunk_config("v2"), FakeEmbedder()
    build_index(cfg, emb, chroma_path=tmp / "chroma", progress=lambda *_: None)
    retr = Retriever(cfg, embedder=emb, reranker=FakeReranker(), chroma_path=tmp / "chroma").ensure_ready()
    qs = metrics.load_questions(ROOT / "tests/fixtures/smoke_questions.jsonl")
    return {m: summarize_retrieval(eval_retrieval(qs, retr, m))["all"] for m in
            ("dense", "bm25", "hybrid", "hybrid_rerank")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args()
    res = run()
    for m, a in res.items():
        print(f"{m:14s} R@1={a['recall@1']:.2f} R@5={a['recall@5']:.2f} MRR={a['mrr']:.2f}")
    got = res[GATE_MODE]["recall@5"]
    if args.update:
        BASELINE.write_text(json.dumps({"recall@5": got, "mode": GATE_MODE}, indent=1))
        print(f"baseline updated: {got}")
        return 0
    base = json.loads(BASELINE.read_text())["recall@5"]
    ok = got >= base - TOLERANCE
    print(f"gate: {GATE_MODE} recall@5={got:.2f} vs baseline={base:.2f} (tolerance {TOLERANCE}) -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
