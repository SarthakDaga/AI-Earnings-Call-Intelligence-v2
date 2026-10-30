import json

from eval import draft_questions as dq

TEXT = ("Data center revenue was $35.6 billion, a record, up 16% sequentially and 93% year on year, as the Blackwell "
        "ramp commenced and Hopper continued sequential growth across every customer segment we serve. " * 2)


class StubLLM:
    def __init__(self, question):
        self.question = question

    def chat(self, messages, **kw):
        return json.dumps({"question": self.question, "gold_answer": "$35.6 billion",
                           "span": "Data center revenue was $35.6 billion, a record, up 16% sequentially and 93% "
                                   "year on year"})


def _setup(monkeypatch):
    chunks = [{"ticker": "NVDA", "quarter": "Q4", "year": 2025, "text": TEXT, "role": "executive"},
              {"ticker": "NVDA", "quarter": "Q4", "year": 2025, "text": TEXT, "role": "operator"},
              {"ticker": "AAPL", "quarter": "Q1", "year": 2026, "text": TEXT, "role": "executive"}]
    monkeypatch.setattr(dq.queries, "get_chunks", lambda h: chunks)
    monkeypatch.setattr(dq.queries, "get_all_transcripts", lambda: [
        {"ticker": "NVDA", "quarter": "Q4", "year": 2025, "date": "2025-02-26"},
        {"ticker": "AAPL", "quarter": "Q1", "year": 2026, "date": "2026-01-29"}])


def test_draft_keeps_questions_that_name_company_and_date(monkeypatch):
    _setup(monkeypatch)
    q = "On the NVIDIA earnings call of 2025-02-26, what was data center revenue?"
    kept = dq.draft("single_fact", 1, 1, tickers={"NVDA"}, llm=StubLLM(q), progress=lambda *_: None)
    assert len(kept) == 1 and kept[0]["id"] == "sf001" and kept[0]["category"] == "single_fact"
    assert kept[0]["evidence"][0]["transcript"] == "NVDA_Q4_2025" and kept[0]["split"] in ("dev", "test")


def test_draft_rejects_questions_without_company_or_date_or_asking_who(monkeypatch):
    _setup(monkeypatch)
    for bad in ("What was data center revenue?",
                "On the NVIDIA call, what was data center revenue?",
                "On the NVIDIA earnings call of 2025-02-26, who said revenue was a record?",
                "According to the passage from the NVIDIA call of 2025-02-26, what was revenue?"):
        kept = dq.draft("single_fact", 1, 1, tickers={"NVDA"}, llm=StubLLM(bad), progress=lambda *_: None)
        assert kept == [], bad


def test_interleave_skips_operator_chunks_and_covers_each_company(monkeypatch):
    _setup(monkeypatch)
    chunks = dq.queries.get_chunks("x")
    out = dq.interleave(chunks, seed=1)
    assert all(c["role"] != "operator" for c in out)
    assert {c["ticker"] for c in out} == {"NVDA", "AAPL"}


def test_cut_off_nvda_passage_is_not_used():
    assert not dq.usable({"text": "it's cold in space " + "word " * 80, "role": "executive"})


def test_merge_moves_spans_to_private_file_and_renumbers(tmp_path):
    from eval import merge_candidates as mc
    cand = tmp_path / "cand.jsonl"
    cand.write_text("\n".join(json.dumps(c) for c in [
        {"id": "sf001", "category": "single_fact", "question": "Q one?", "gold_answer": "A1", "split": "dev",
         "needs_human_review": True, "evidence": [{"transcript": "NVDA_Q4_2025", "span": "verbatim one"}]},
        {"id": "sf002", "category": "single_fact", "question": "Q two?", "gold_answer": "A2", "split": "test",
         "needs_human_review": True, "evidence": [{"transcript": "AAPL_Q1_2026", "span": "verbatim two"}]}]) + "\n")
    questions, private = tmp_path / "questions.jsonl", tmp_path / "evidence.private.jsonl"
    questions.write_text('# starter\n{"id":"q040","category":"unanswerable","question":"Meta?","gold_answer":"",'
                         '"evidence":[],"answerable":false,"split":"dev"}\n')
    ids = mc.merge(cand, ["sf002", "sf001"], questions, private)
    assert ids == ["q101", "q102"]
    qs = [json.loads(x) for x in questions.read_text().splitlines() if x and not x.startswith("#")]
    assert qs[-2]["id"] == "q101" and qs[-2]["question"] == "Q two?" and qs[-2]["evidence"] == []
    assert "verbatim" not in questions.read_text() and "needs_human_review" not in questions.read_text()
    ev = [json.loads(x) for x in private.read_text().splitlines()]
    assert ev[0] == {"id": "q101", "evidence": [{"transcript": "AAPL_Q1_2026", "span": "verbatim two"}]}
    assert mc.merge(cand, ["sf001"], questions, private) == []  # the same question is never merged twice
