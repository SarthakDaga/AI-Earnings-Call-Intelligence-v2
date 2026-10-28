import json

import pytest

from eval import metrics
from eval.judges import Judge
from eval.run_eval import (
    aggregate_answers,
    eval_answers,
    eval_retrieval,
    make_runner,
    render_answers,
    render_retrieval,
    summarize_retrieval,
)
from eval.verify_evidence import verify
from src.llm import FakeLLM

FIX = "tests/fixtures/transcripts"


def test_span_matching_handles_number_formats_and_splits():
    chunk = "Operating expenses were $1,350,000,000 for the quarter."
    assert metrics.span_in_chunk(chunk, "Operating expenses were $1.35 billion")
    assert not metrics.span_in_chunk(chunk, "Operating expenses were $9 billion for the quarter")
    long_span = "we qualified a second vendor and expect constraints to ease by the third quarter okay"
    assert metrics.span_in_chunk("we qualified a second vendor and expect constraints to ease by the third", long_span)


def test_hit_requires_matching_transcript():
    hit = {"ticker": "ACME", "quarter": "Q2", "year": 2025, "text": "Tariffs cost roughly one point of margin."}
    ev = {"transcript": "ACME_Q1_2025", "span": "Tariffs cost roughly one point of margin."}
    assert not metrics.hit_matches(hit, ev)
    assert metrics.hit_matches(hit, {**ev, "transcript": "ACME_Q2_2025"})


def test_score_retrieval_values():
    hits = [{"ticker": "A", "quarter": "Q1", "year": 2025, "text": "nothing here at all"},
            {"ticker": "A", "quarter": "Q1", "year": 2025, "text": "margin was 52 percent this quarter overall"}]
    ev = [{"transcript": "A_Q1_2025", "span": "margin was 52 percent this quarter overall"},
          {"transcript": "A_Q1_2025", "span": "a second span that is never retrieved anywhere"}]
    s = metrics.score_retrieval(hits, ev)
    assert s["hit@1"] == 0 and s["hit@3"] == 1 and s["rr"] == 0.5 and s["cov@3"] == 0.5


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    vals = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    lo, hi = metrics.bootstrap_ci(vals)
    assert lo <= 0.7 <= hi and metrics.bootstrap_ci(vals) == [lo, hi]
    assert metrics.bootstrap_ci([]) is None


def test_numeric_exact():
    assert metrics.numeric_exact_ok("$4.5 billion", "Guidance was $4,500,000,000 for Q2")
    assert not metrics.numeric_exact_ok("$4.5 billion", "Guidance was $5 billion")
    assert metrics.numeric_exact_ok("no numbers here", "anything")


def test_load_questions_merges_private_evidence(tmp_path):
    (tmp_path / "questions.jsonl").write_text(
        '# comment\n{"id":"a","category":"single_fact","question":"q?","gold_answer":"x"}\n'
        '{"id":"b","category":"cross_quarter","question":"q2?","split":"test"}\n')
    (tmp_path / "evidence.private.jsonl").write_text(
        json.dumps({"id": "a", "evidence": [{"transcript": "T_Q1_2025", "span": "s"}]}) + "\n")
    qs = metrics.load_questions(tmp_path / "questions.jsonl")
    assert qs[0]["evidence"] and qs[1]["agent_subset"] and qs[0]["answerable"]
    assert len(metrics.load_questions(tmp_path / "questions.jsonl", "test")) == 1


def test_missing_questions_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="eval/README"):
        metrics.load_questions(tmp_path / "nope.jsonl")


def test_retrieval_eval_end_to_end(retriever):
    qs = metrics.load_questions("tests/fixtures/smoke_questions.jsonl")
    rows = eval_retrieval(qs, retriever, "hybrid")
    assert len(rows) == 7  # unanswerable question skipped
    s = summarize_retrieval(rows)
    assert s["all"]["recall@5"] >= 0.8 and "single_fact" in s["by_category"]
    assert s["all"]["ci95_recall@5"] and s["dev"]["n"] + s["test"]["n"] == 7
    assert "A3" not in render_retrieval({"rows": {}})  # renders with no rows


def test_verify_evidence_catches_bad_span(tmp_path):
    qs = metrics.load_questions("tests/fixtures/smoke_questions.jsonl")
    assert verify(qs, __import__("pathlib").Path(FIX)) == []
    bad = [{"id": "x", "category": "single_fact", "answerable": True,
            "evidence": [{"transcript": "ACME_Q1_2025", "span": "this sentence is not in the file"}]}]
    assert any("span not found" in e for e in verify(bad, __import__("pathlib").Path(FIX)))


def test_judge_caches_and_handles_garbage(tmp_path):
    llm = FakeLLM(['{"claims":[{"claim":"a","supported":true},{"claim":"b","supported":false}]}'])
    j = Judge(llm, tmp_path)
    assert j.faithfulness("ans", "ctx") == 0.5
    assert j.faithfulness("ans", "ctx") == 0.5 and len(llm.calls) == 1  # second served from cache
    assert Judge(FakeLLM(["not json"]), tmp_path / "b").faithfulness("a", "c") is None
    assert Judge(FakeLLM(['{"score": 0.4}']), tmp_path / "c").correctness("q", "g", "a") == 0.5


def test_answer_eval_rag_end_to_end(retriever):
    qs = metrics.load_questions("tests/fixtures/smoke_questions.jsonl")
    script = lambda m: ("Gross margin was 52 percent [c1]." if "gross margin" in m[-1]["content"].split("Question:")[-1].lower()  # noqa: E731
                        else "INSUFFICIENT_EVIDENCE")
    judge = Judge(FakeLLM(lambda m: '{"claims":[{"claim":"x","supported":true}]}' if "fact-checker" in m[0]["content"]
                          else '{"score": 1, "reason": "ok"}'), __import__("pathlib").Path("/tmp/judge_cache_test"))
    runner = make_runner(dict(kind="rag", mode="hybrid_rerank"), retriever, FakeLLM(script))
    rows = eval_answers(qs, runner, judge)
    by = {r["id"]: r for r in rows}
    assert by["s1"]["correctness"] == 1.0 and by["s1"]["faithfulness"] == 1.0
    assert by["s8"]["abstained"]                       # unanswerable handled
    assert by["s2"]["correctness"] == 0.0 and by["s2"]["abstained"]  # abstained on answerable => 0
    agg = aggregate_answers(rows)
    assert agg["abstention"] == 1.0 and 0 < agg["correctness"] < 1
    assert "B0" not in render_answers({"rows": {}})


def test_tables_render_into_readme(tmp_path, monkeypatch):
    from eval import run_eval
    from src import config
    (tmp_path / "results").mkdir()
    (tmp_path / "README.md").write_text("x\n<!-- RETRIEVAL_TABLE_START -->\nold\n<!-- RETRIEVAL_TABLE_END -->\n"
                                        "<!-- ANSWER_TABLE_START -->\nold\n<!-- ANSWER_TABLE_END -->\n")
    monkeypatch.setattr(run_eval, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    agg = {"recall@1": .5, "recall@3": .7, "recall@5": .8, "recall@10": .9, "mrr": .65, "p50_ms": 12.0}
    run_eval.save("retrieval", {"A0": {"summary": {"all": agg, "by_category": {"single_fact": agg}}}}, "all", {})
    ans = {"faithfulness": .9, "correctness": .6, "citation_precision": .7, "abstention": 1.0, "p50_ms": 2500.0}
    run_eval.save("answers", {"B1": {"summary": ans}}, "all", {})
    run_eval.update_readme()
    text = (tmp_path / "README.md").read_text()
    assert "| A0 Baseline" in text and "0.80" in text and "100%" in text and "old" not in text
    assert "single_fact R@5" in (tmp_path / "results" / "latest.md").read_text()


def test_cli_without_question_set_exits_cleanly(tmp_path):
    from eval import run_eval
    with pytest.raises(SystemExit) as e:
        run_eval.main(["retrieval", "--questions", str(tmp_path / "missing.jsonl")])
    assert "questions" in str(e.value)
