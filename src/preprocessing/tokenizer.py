"""
src/preprocessing/tokenizer.py
Sentence tokenization (NLTK Punkt) and the v1 word-window chunker (kept for A0).
"""

import nltk


def _ensure_nltk():
    for res, pkg in (("tokenizers/punkt_tab", "punkt_tab"), ("tokenizers/punkt", "punkt")):
        try:
            nltk.data.find(res)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:  # offline: fall back to regex splitter below
                pass


_ensure_nltk()


def tokenize_sentences(text: str) -> list[str]:
    """Split into sentences; drop fragments shorter than 10 chars."""
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) >= 10]


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """v1 word-window chunking (baseline A0)."""
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks
