"""
src/retrieval/embedder.py
sentence-transformers wrapper (free, local). `FakeEmbedder` is a deterministic
hashing bag-of-words embedder used by tests and CI so no model download is needed.
"""

from __future__ import annotations

import hashlib
import re
import threading

import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()  # concurrent first requests must share ONE model load

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    print(f"[Embedder] Loading '{self.model_name}'...")
                    self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def max_seq_length(self) -> int:
        return int(self._load().max_seq_length)

    def encode_docs(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1))
        return self._load().encode(texts, batch_size=batch_size, show_progress_bar=show_progress,
                                   convert_to_numpy=True, normalize_embeddings=True)

    def encode_query(self, text: str) -> np.ndarray:
        if "bge" in self.model_name.lower():
            text = BGE_QUERY_PREFIX + text
        return self.encode_docs([text])[0]

    def token_counts(self, texts: list[str]) -> list[int]:
        tok = self._load().tokenizer
        return [len(tok.encode(t, add_special_tokens=True)) for t in texts]


class FakeEmbedder:
    """Hashing bag-of-words, L2-normalized. Deterministic, dependency-free."""
    model_name = "fake-hash"
    loaded = True
    max_seq_length = 256

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        for tok in re.findall(r"[a-z0-9$%.]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_docs(self, texts, batch_size=32, show_progress=False):
        return np.array([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def encode_query(self, text):
        return self._vec(text)

    def token_counts(self, texts):
        return [len(t.split()) for t in texts]
