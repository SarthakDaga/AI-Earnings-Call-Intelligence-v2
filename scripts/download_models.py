"""Pre-download free open models (used at Docker build time so the container starts fast)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finbert", action="store_true", help="also fetch FinBERT (only needed offline for sentiment)")
    ap.add_argument("--base-reranker", action="store_true")
    a = ap.parse_args()
    from sentence_transformers import CrossEncoder, SentenceTransformer
    SentenceTransformer(config.EMBED_MODEL)
    CrossEncoder(config.RERANKER_SMALL)
    if a.base_reranker:
        CrossEncoder(config.RERANKER_BASE)
    if a.finbert:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        AutoTokenizer.from_pretrained(config.FINBERT_MODEL)
        AutoModelForSequenceClassification.from_pretrained(config.FINBERT_MODEL)
    print("models cached")


if __name__ == "__main__":
    main()
