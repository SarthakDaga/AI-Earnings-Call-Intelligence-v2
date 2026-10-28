"""Cross-encoder reranker (R2.4). Free open models; CPU is fine for 30 candidates."""

from __future__ import annotations

import re
import threading


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def loading(self) -> bool:
        """True while another thread is loading/downloading the model."""
        return self._model is None and self._lock.locked()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder
                    print(f"[Reranker] Loading '{self.model_name}'...")
                    self._model = CrossEncoder(self.model_name, max_length=512)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._load().predict([(query, t) for t in texts])]


class FakeReranker:
    """Token-overlap scorer for tests."""
    model_name = "fake-overlap"
    loaded = True

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(re.findall(r"[a-z0-9$%.]+", query.lower()))
        return [len(q & set(re.findall(r"[a-z0-9$%.]+", t.lower()))) / (len(q) or 1) for t in texts]
