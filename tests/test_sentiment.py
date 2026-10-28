"""Aggregation + tokenizer logic (no model loading)."""

from src.preprocessing.tokenizer import chunk_text, tokenize_sentences
from src.sentiment.finbert import aggregate_by_section, aggregate_sentiment, run_sentiment_pipeline


class TestAggregateSentiment:
    def _make_scores(self, labels):
        probs = {"positive": {"positive": 0.9, "negative": 0.05, "neutral": 0.05},
                 "negative": {"positive": 0.05, "negative": 0.9, "neutral": 0.05},
                 "neutral": {"positive": 0.1, "negative": 0.1, "neutral": 0.8}}
        return [{"text": f"Sentence {i}", "label": lab, "score": probs[lab][lab], **probs[lab]}
                for i, lab in enumerate(labels)]

    def test_all_positive(self):
        r = aggregate_sentiment(self._make_scores(["positive"] * 10))
        assert r["positive_pct"] == 100.0 and r["negative_pct"] == 0.0 and r["net_score"] > 0

    def test_all_negative(self):
        r = aggregate_sentiment(self._make_scores(["negative"] * 10))
        assert r["negative_pct"] == 100.0 and r["net_score"] < 0

    def test_mixed_sentiment(self):
        r = aggregate_sentiment(self._make_scores(["positive"] * 5 + ["negative"] * 3 + ["neutral"] * 2))
        assert (r["positive_pct"], r["negative_pct"], r["neutral_pct"]) == (50.0, 30.0, 20.0)

    def test_top_sentences_capped_at_5(self):
        r = aggregate_sentiment(self._make_scores(["positive"] * 10))
        assert len(r["top_positive_sentences"]) <= 5 and len(r["top_negative_sentences"]) == 0

    def test_empty_input(self):
        assert aggregate_sentiment([]) == {}

    def test_net_score_range(self):
        for labels in (["positive"] * 10, ["negative"] * 10, ["neutral"] * 10,
                       ["positive"] * 4 + ["negative"] * 3 + ["neutral"] * 3):
            assert -1.0 <= aggregate_sentiment(self._make_scores(labels))["net_score"] <= 1.0


def test_aggregate_by_section_and_runner_with_fake_scorer():
    def fake(sentences):
        return [{"text": s, "label": "positive" if "good" in s else "negative", "score": .9,
                 "positive": .9 if "good" in s else .05, "negative": .05 if "good" in s else .9,
                 "neutral": .05} for s in sentences]
    _, scores = run_sentiment_pipeline(["good news here", "bad news here"], ["prepared", "qa"], scorer=fake)
    assert scores[1]["section"] == "qa"
    assert aggregate_by_section(scores, "prepared")["net_score"] > 0
    assert aggregate_by_section(scores, "qa")["net_score"] < 0
    assert aggregate_by_section(scores, "missing") == {}


class TestTokenizer:
    def test_sentence_tokenization(self):
        s = tokenize_sentences("Revenue grew 40 percent. We saw strong demand across all segments. "
                               "Margins expanded significantly.")
        assert len(s) == 3

    def test_filters_short_sentences(self):
        s = tokenize_sentences("Hi. Revenue grew 40 percent year over year driven by AI demand.")
        assert all(len(x) >= 10 for x in s)

    def test_chunk_text_basic(self):
        chunks = chunk_text(" ".join(["word"] * 1000), chunk_size=400, overlap=50)
        assert len(chunks) > 1 and all(len(c.split()) <= 400 for c in chunks)

    def test_chunk_overlap(self):
        chunks = chunk_text(" ".join(f"word{i}" for i in range(500)), chunk_size=200, overlap=50)
        assert set(chunks[0].split()[-50:]) & set(chunks[1].split()[:50])

    def test_empty_text(self):
        assert chunk_text("") == [] and tokenize_sentences("") == []
