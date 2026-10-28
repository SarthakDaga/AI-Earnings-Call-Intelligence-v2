"""Reciprocal Rank Fusion (R2.3). score(d) = sum over lists of weight / (k + rank)."""


def rrf(rankings: list[list[str]], k: int = 60, weights: list[float] | None = None) -> list[tuple[str, float]]:
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
