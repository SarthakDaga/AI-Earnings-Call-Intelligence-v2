"""
src/sentiment/finbert.py

FinBERT financial sentiment (ProsusAI/finbert, free, CPU-friendly).
torch/transformers are imported lazily so the API server, tests and the agent
never pay for (or need) them: sentiment is precomputed offline (F7).
"""

from src import config

MODEL_NAME = config.FINBERT_MODEL
LABELS = ["positive", "negative", "neutral"]  # FinBERT output order

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _tokenizer is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        print(f"[FinBERT] Loading model '{MODEL_NAME}'... (first run may take a minute)")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def score_sentences(sentences: list[str], batch_size: int = 16) -> list[dict]:
    """Run FinBERT in batches. Returns [{text,label,score,positive,negative,neutral}]."""
    if not sentences:
        return []
    import torch
    from torch.nn.functional import softmax
    from tqdm import tqdm

    tokenizer, model = _load_model()
    results = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="[FinBERT]", unit="batch"):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1).numpy()
        for sentence, prob in zip(batch, probs):
            pos, neg, neu = float(prob[0]), float(prob[1]), float(prob[2])
            results.append({"text": sentence, "label": LABELS[prob.argmax()],
                            "score": float(prob.max()), "positive": pos,
                            "negative": neg, "neutral": neu})
    return results


def aggregate_sentiment(sentence_scores: list[dict]) -> dict:
    """net = avg(positive) - avg(negative), range -1..+1."""
    if not sentence_scores:
        return {}
    total = len(sentence_scores)
    count = lambda lab: sum(1 for s in sentence_scores if s["label"] == lab)  # noqa: E731
    avg_pos = sum(s["positive"] for s in sentence_scores) / total
    avg_neg = sum(s["negative"] for s in sentence_scores) / total
    top = lambda lab: sorted([s for s in sentence_scores if s["label"] == lab],  # noqa: E731
                             key=lambda x: x["score"], reverse=True)[:5]
    return {
        "positive_pct": round(count("positive") / total * 100, 1),
        "neutral_pct": round(count("neutral") / total * 100, 1),
        "negative_pct": round(count("negative") / total * 100, 1),
        "net_score": round(avg_pos - avg_neg, 4),
        "top_positive_sentences": [{"text": s["text"], "score": s["score"]} for s in top("positive")],
        "top_negative_sentences": [{"text": s["text"], "score": s["score"]} for s in top("negative")],
    }


def aggregate_by_section(sentence_scores: list[dict], section: str | None = None) -> dict:
    """Aggregate only 'prepared' or 'qa' sentences (rows scored before v2 have no section)."""
    if section is None:
        return aggregate_sentiment(sentence_scores)
    subset = [s for s in sentence_scores if s.get("section") == section]
    return aggregate_sentiment(subset)


def run_sentiment_pipeline(sentences: list[str], sections: list[str] | None = None,
                           scorer=None) -> tuple[dict, list[dict]]:
    """sentences -> (aggregated metrics, per-sentence scores incl. section)."""
    scorer = scorer or score_sentences
    scores = scorer(sentences)
    if sections:
        for s, sec in zip(scores, sections):
            s["section"] = sec
    return aggregate_sentiment(scores), scores
